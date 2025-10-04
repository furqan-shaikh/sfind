import asyncio

import torch

from sfind.models.models import (
    FetchCaptionRequest,
    FetchCaptionResponse
)
from sfind.model_engine.utils import get_device, get_file_data
from transformers import BlipProcessor, BlipForConditionalGeneration


class SalesForceBlipBaseModelBackEnd:
    def __init__(self):
        self.model_id = "Salesforce/blip-image-captioning-base"
        self.device = get_device()
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = (BlipForConditionalGeneration.from_pretrained(self.model_id,
                                                             torch_dtype=torch.float16)
                 .to(self.device))
    def get_model_id(self) -> str:
        return self.model_id

    async def get_caption(self, caption_request: FetchCaptionRequest) -> FetchCaptionResponse:
        def _infer():

            raw_image = get_file_data(input_data=caption_request.file_data)
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device, torch.float16)
            out = self.model.generate(**inputs)
            output = self.processor.decode(out[0], skip_special_tokens=True)
            return FetchCaptionResponse(caption=output)

        return await asyncio.to_thread(_infer)

