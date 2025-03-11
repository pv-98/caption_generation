

# Setup

Follow the steps to create a virtual environment and install the updated packages mentioned in requirements.txt file

```
$ python3 -m venv ally_env
$ source ally_env/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```

# Environment Variables

Set the API keys given in the assignment description (zip folder)

# Use the following command to run the assistant

```
$ python3 assistant.py start
```

# Solution Description

To Detect the person in an image, I used faster-rcnn object detaction model with resnet-50 as the backbone.
If the user prompt is image related, the assistant runs the object detection model and search for a person.
If a person is detected, A series of steps is performed
- Image retrieval and Conversion: The frames captured from the video track are extracted and selects a best frame. This is then converted to an image format suitable for processing.
- Person Detection Check: When ``` process_image_question ``` is invoked with ``` use_image = True ```, the best frame is detected and then calls ``` detect_person_in_frame``` to check for person detection.
- BLIP Caption Generation: If a person is detected, the assistant uses BLIP model to generate an initial caption through its vision-language model.
- GPT-2 for Description: GPT-2 model is used to provide a detailed description based on the generated caption.

- P.S: Larger language models could have been used like Falcon provided by Hugging face. But due to the high processing power required (Powerful GPUs), which is unavailable at the moment, the approach of combining BLIP and GPT-2 has been taken. Otherwise, an integrated LLM model with multimodal capabilities can be used.
