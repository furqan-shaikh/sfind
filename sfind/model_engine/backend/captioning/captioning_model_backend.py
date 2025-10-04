from typing import Protocol

from sfind.models.models import (
    FetchCaptionRequest,
    FetchCaptionResponse
)


class CaptioningModelBackEnd(Protocol):
    def get_model_id(self) -> str: ...
    async def get_caption(self, caption_request: FetchCaptionRequest) -> FetchCaptionResponse:  ...

