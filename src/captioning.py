from io import BytesIO

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cpu")
img_url = '/Users/furqanshaikh/Documents/dev/sfind/images/tennis_2.jpg'

def do_conditional_captioning():
    raw_image = Image.open(img_url).convert('RGB')

    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cpu", torch.float16)
    #
    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

def do_unconditional_captioning():
    with open(img_url, "rb") as f:
        image_bytes = f.read()
    raw_image = Image.open(BytesIO(image_bytes)).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to("cpu", torch.float16)
    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

do_unconditional_captioning()
