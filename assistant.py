import asyncio
from typing import Annotated
import io
import concurrent.futures
import os  # Import os for environment variables
import torch
import torchvision
from PIL import Image
import cv2
import numpy as np

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero, elevenlabs

from transformers import AutoTokenizer, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration


# ---------------- Device Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ---------------- Person Detection Setup ----------------
person_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
person_model.eval()
person_model.to(device)

# ---------------- BLIP Image Captioning Setup ----------------
try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model.eval()
    blip_model.to(device)
    # if torch.cuda.is_available():
    #     blip_model.half()
    print("[INFO] BLIP model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load BLIP model: {e}")
    blip_processor = None
    blip_model = None

# ---------------- GPT-2 Model Setup ----------------
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_model.eval()
    gpt2_model.to(device)
    # if torch.cuda.is_available():
    #     gpt2_model.half()
    print("[INFO] GPT-2 model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load GPT-2 model: {e}")
    tokenizer = None
    gpt2_model = None

executor = concurrent.futures.ThreadPoolExecutor()

def detect_person_in_frame(image_bytes: bytes) -> bool:
    """
    Detect if a person is present in the image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        
        image_tensor = transform(image).to(device)
        with torch.no_grad():
            predictions = person_model([image_tensor])
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        for label, score in zip(labels, scores):
            if label == 1 and score > 0.5:
                return True
        return False
    except Exception as e:
        print(f"[ERROR] detect_person_in_frame: {e}")
        return False

def generate_image_caption(image: Image.Image) -> str:
    """
    Use BLIP to caption the image.
    """
    if blip_processor is None or blip_model is None:
        return "Image captioning is currently unavailable."
    try:
        inputs = blip_processor(image, return_tensors="pt").to(device)
        # if torch.cuda.is_available():
        #     inputs = {k: v.half() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = blip_model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"[ERROR] generate_image_caption: {e}")
        return "Unable to generate a caption."

def generate_gpt2_response(prompt: str) -> str:
    """
    Generate a response using GPT-2.
    """
    if tokenizer is None or gpt2_model is None:
        return "Text generation is currently unavailable."
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gpt2_model.generate(
                **inputs,
                max_new_tokens=50,  # Generate up to 50 new tokens
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True
            )
        response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"[ERROR] generate_gpt2_response: {e}")
        return "Sorry, I couldn't generate a response at this time."

async def generate_gpt2_response_async(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    print(f"[LOG] Generating GPT-2 response for prompt: {prompt}")
    try:
        response = await loop.run_in_executor(executor, generate_gpt2_response, prompt)
        print(f"[LOG] Generated GPT-2 response: {response}")
        return response
    except Exception as e:
        print(f"[ERROR] Error generating GPT-2 response: {e}")
        return "Sorry, I couldn't generate a response at this time."

class AssistantFunction(agents.llm.FunctionContext):
    """Functions callable by the LLM."""
    @agents.llm.ai_callable(description="Evaluate something requiring vision capabilities")
    async def image(
        self,
        user_msg: Annotated[str, agents.llm.TypeInfo(description="The user message")]
    ):
        print(f"[LOG] Vision capability triggered by user message: {user_msg}")
        return None

async def single_response_generator(response: str):
    yield response

async def stream_gpt2_response(prompt: str):
    response = await generate_gpt2_response_async(prompt)
    print(f"[LOG] Full generated response: {response}")
    for token in response.split():
        print(f"[LOG] Streaming token: {token}")
        yield token + ' '
        await asyncio.sleep(0)

async def get_video_track(room: rtc.Room):
    video_track = asyncio.Future()
    for participant in room.remote_participants.values():
        for pub in participant.track_publications.values():
            if (pub.track and pub.track.kind == rtc.TrackKind.KIND_VIDEO and
                isinstance(pub.track, rtc.RemoteVideoTrack)):
                video_track.set_result(pub.track)
                return await video_track

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if not video_track.done() and track.kind == rtc.TrackKind.KIND_VIDEO and isinstance(track, rtc.RemoteVideoTrack):
            video_track.set_result(track)
    try:
        return await asyncio.wait_for(video_track, timeout=10.0)
    except asyncio.TimeoutError:
        print("[ERROR] Timeout waiting for video track")
        raise Exception("No video track received")

async def _enableCamera(ctx):
    await ctx.room.local_participant.publish_data("camera_enable", reliable=True, topic="camera")


def video_frame_to_pil_image(frame) -> Image.Image:
    """
    Convert a VideoFrame object to a PIL.Image.Image.

    Parameters:
        frame: The VideoFrame object.

    Returns:
        Image.Image: The converted PIL Image or None if conversion fails.
    """
    try:
        
        y_bytes = frame.get_plane(0).tobytes()
        u_bytes = frame.get_plane(1).tobytes()
        v_bytes = frame.get_plane(2).tobytes()

        height = frame.height
        width = frame.width

        # I420 format (YUV 4:2:0)
        i420 = y_bytes + u_bytes + v_bytes

        # Convert I420 to BGR
        i420_frame = np.frombuffer(i420, dtype=np.uint8).reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(i420_frame, cv2.COLOR_YUV2BGR_I420)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        image_pil = Image.fromarray(rgb)
        return image_pil
    except Exception as e:
        print(f"[ERROR] video_frame_to_pil_image: {e}")
        return None


def compute_sharpness(image: Image.Image) -> float:
    """
    Compute the sharpness of an image using the variance of the Laplacian.
    
    Parameters:
        image (PIL.Image.Image): The image to evaluate.
        
    Returns:
        float: Variance of Laplacian, representing sharpness. Higher is sharper.
    """
    try:
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance
    except Exception as e:
        print(f"[ERROR] compute_sharpness: {e}")
        return 0.0

async def select_best_frame(latest_images_deque):
    """
    Select the best frame from a deque of frames based on sharpness.

    Parameters:
        latest_images_deque (list): List of VideoFrame objects.

    Returns:
        VideoFrame: The frame with the highest sharpness, or None if none found.
    """
    if not latest_images_deque:
        print("[WARNING] No frames available to select from.")
        return None

    loop = asyncio.get_event_loop()
    best_sharpness = -1.0
    best_frame = None

    # Convert frames to images and compute sharpness asynchronously
    for frame in latest_images_deque:
        # Convert VideoFrame to PIL Image
        image_pil = video_frame_to_pil_image(frame)
        if image_pil is None:
            continue

        # Compute sharpness
        sharpness = compute_sharpness(image_pil)
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame = frame

    if best_frame is None:
        print("[WARNING] No valid frames after sharpness computation.")
        return None

    print(f"[INFO] Selected best frame with sharpness {best_sharpness:.2f}")
    return best_frame

async def _getVideoFrame(ctx, assistant):
    await _enableCamera(ctx)
    latest_images_deque = []
    try:
        print("[LOG] Waiting for video track...")
        video_track = await get_video_track(ctx.room)
        print(f"[LOG] Got video track: {video_track.sid}")
        async for event in rtc.VideoStream(video_track):
            latest_frame = event.frame
            latest_images_deque.append(latest_frame)
            assistant.fnc_ctx.latest_image = latest_frame
            if len(latest_images_deque) == 5:
                # Use the updated select_best_frame function
                best_frame = await select_best_frame(latest_images_deque)
                if best_frame is None:
                    print("[ERROR] No best frame selected.")
                    return None

                # Convert best_frame to JPEG bytes
                image_pil = video_frame_to_pil_image(best_frame)
                if image_pil is None:
                    print("[ERROR] Failed to convert best frame to PIL image.")
                    return None

                buffer = io.BytesIO()
                image_pil.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
                return image_bytes
    except Exception as e:
        print(f"[ERROR] Error in getVideoFrame function: {e}")
        return None

async def proprietary_model_describe(chat_image: ChatImage = None) -> str:
    if gpt is None:
        return "Sorry, proprietary model unavailable."
    # **FIX:** Ensure content is a string, not a ChatImage object
    if chat_image is not None:
        # If chat_image is provided, you might process it as needed
        chat_context.messages.append(ChatMessage(role="user", content="User provided an image."))
    else:
        chat_context.messages.append(ChatMessage(role="user", content="User did not provide an image."))
    try:
        stream = gpt.chat(chat_ctx=chat_context)
        response = ""
        async for token in stream:
            response += token
            if len(response.split()) >= 200:
                break
        return response.strip()
    except Exception as e:
        print(f"[ERROR] proprietary_model_describe: {e}")
        return "Sorry, I couldn't describe the image."

async def proprietary_model_response(text: str) -> str:
    if gpt is None:
        return "Sorry, proprietary model unavailable."
    chat_context.messages.append(ChatMessage(role="user", content=text))
    try:
        stream = gpt.chat(chat_ctx=chat_context)
        response = ""
        async for token in stream:
            response += token
            if len(response.split()) >= 200:
                break
        return response.strip()
    except Exception as e:
        print(f"[ERROR] proprietary_model_response: {e}")
        return "Sorry, I couldn't process your request."

semaphore = asyncio.Semaphore(2)

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"[INFO] Connected to room: {ctx.room.name}")

    global gpt, chat_context
    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Ally. You are an assistant for the blind and visually impaired. "
                    "You respond with short and concise answers. Avoid using unpronounceable punctuation or emojis. "
                    "If the user requests image understanding, you may call the 'image' function."
                ),
            )
        ]
    )

    # Proprietary model
    try:
        gpt = openai.LLM(model="gpt-4o")
        print("[INFO] Proprietary GPT model initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize proprietary GPT model: {e}")
        gpt = None

    # Custom TTS Voice
    try:
        custom_voice = elevenlabs.Voice(
            id='21m00Tcm4TlvDq8ikWAM',
            name='Bella',
            category='premade',
            settings=elevenlabs.VoiceSettings(
                stability=0.71,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            )
        )
        elevenlabs_tts = elevenlabs.TTS(voice=custom_voice)
        print("[INFO] ElevenLabs TTS initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize ElevenLabs TTS: {e}")
        elevenlabs_tts = None

    # ---------------- Retrieve Deepgram API Key from Environment ----------------
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    if not DEEPGRAM_API_KEY:
        print("[ERROR] DEEPGRAM_API_KEY environment variable not set.")
        return

    try:
        assistant = VoiceAssistant(
            vad=silero.VAD.load(),
            stt=deepgram.STT(api_key=DEEPGRAM_API_KEY),
            llm=gpt,
            tts=elevenlabs_tts,
            fnc_ctx=AssistantFunction(),
            chat_ctx=chat_context,
        )
        print("[INFO] VoiceAssistant initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize VoiceAssistant: {e}")
        return

    chat = rtc.ChatManager(ctx.room)
    assistant.start(ctx.room)

    # Greet the user immediately after assistant is ready
    await assistant.say("Hi there! How can I help?", allow_interruptions=True)

    async def process_image_question(text: str, use_image: bool = False):
        async with semaphore:
            if use_image:
                print("[LOG] Getting video frame")
                latest_image_bytes = await _getVideoFrame(ctx, assistant)
                if latest_image_bytes is not None:
                    print("[LOG] Checking for person in the image")
                    if detect_person_in_frame(latest_image_bytes):
                        print("[LOG] Person detected! Using BLIP and GPT-2 for description")
                        try:
                            image = Image.open(io.BytesIO(latest_image_bytes)).convert("RGB")
                        except Exception as e:
                            print(f"[ERROR] Failed to open image: {e}")
                            return "Sorry, I couldn't process the image."
                        base_caption = generate_image_caption(image)
                        print(f"[LOG] Base caption generated: {base_caption}")
                        prompt = f"Given the following image caption:\nImage Caption: {base_caption}\nDescription:"
                        description = await generate_gpt2_response_async(prompt)
                        return description.strip()
                    else:
                        print("[LOG] No person detected. Using proprietary model")
                        proprietary_description = await proprietary_model_describe()
                        return proprietary_description
                else:
                    print("[LOG] No image available")
                    return "Sorry, I couldn't retrieve an image to describe."
            else:
                # Non-image question: use proprietary model
                proprietary_response = await proprietary_model_response(text)
                return proprietary_response

    async def _answer(text: str, use_image: bool = False):
        print(f"[LOG] _answer called with text: {text}, use_image: {use_image}")
        response = await process_image_question(text, use_image)
        if not response:
            response = "I'm sorry, I couldn't process your request."
        stream = stream_gpt2_response(response)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """
        User sent a message: respond accordingly.
        """
        if msg.message:
            # If user specifically requests image understanding:
            # For simplicity, assume use_image=True if user says 'image'
            use_image = 'image' in msg.message.lower()
            asyncio.create_task(_answer(msg.message, use_image=use_image))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """
        Handle function calls if any are triggered by the model.
        """
        print(f"[LOG] Function calls finished. Number of calls: {len(called_functions)}")
        if len(called_functions) == 0:
            print("[LOG] No functions were called")
            return

        try:
            user_msg = called_functions[0].call_info.arguments.get("user_msg")
            print(f"[LOG] Function call user message: {user_msg}")
            if user_msg:
                print("[LOG] Creating task for _answer with image")
                asyncio.create_task(_answer(user_msg, use_image=True))
            else:
                print("[LOG] No user message to process")
        except Exception as e:
            print(f"[ERROR] Error in function_calls_finished: {e}")

    # Keep running
    await asyncio.sleep(3600)  # Keep the assistant alive for a while, adjust as needed.

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
