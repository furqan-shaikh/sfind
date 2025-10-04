from io import BytesIO
from typing import Union
import torch
from PIL import Image

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_file_data(input_data: Union[str| bytes]):
    if isinstance(input_data, str):
        return Image.open(input_data).convert('RGB')
    elif isinstance(input_data, bytes):
        return Image.open(BytesIO(input_data)).convert('RGB')
    else:
        raise Exception("Invalid file data")