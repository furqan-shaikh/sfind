from typing import Protocol
from sfind.models.models import (
    EmbedTextRequest, EmbedTextResponse,
    EmbedImageRequest, EmbedImageResponse,
    SimilarityScoreResponse, SimilarityScoreRequest, SimilarityScoreUsingSpaceRequest
)

class EncoderModelBackEnd(Protocol):
    """Backend interface for embedding models."""

    async def embed_text(self, embed_text_request: EmbedTextRequest) -> EmbedTextResponse: ...
    async def embed_image(self, embed_image_request: EmbedImageRequest) -> EmbedImageResponse: ...

    async def get_similarity_score(self,
                                   similarity_score_request: SimilarityScoreRequest) -> SimilarityScoreResponse: ...

    async def get_similarity_score_using_space(self, request: SimilarityScoreUsingSpaceRequest) -> SimilarityScoreResponse: ...