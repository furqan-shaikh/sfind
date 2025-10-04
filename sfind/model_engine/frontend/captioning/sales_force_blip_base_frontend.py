from sfind.model_engine.backend.captioning.sales_force_blip_base_backend import SalesForceBlipBaseModelBackEnd
from sfind.models.models import (
    FetchCaptionRequest,
    FetchCaptionResponse
)


class SalesForceBlipBaseModelFrontEnd:
    def __init__(self):
        self.model_back_end = SalesForceBlipBaseModelBackEnd()

    def get_model_id(self) -> str:
        return self.model_back_end.get_model_id()

    async def get_caption(self, caption_request: FetchCaptionRequest) -> FetchCaptionResponse:
        return await self.model_back_end.get_caption(caption_request=caption_request)

