from sfind.model_engine.backend.encoder.perception_encoder_backend import PerceptionEncoderModelBackEnd
from sfind.models.models import (
    EmbedTextRequest, EmbedTextResponse,
    EmbedImageRequest, EmbedImageResponse, SimilarityScoreResponse, SimilarityScoreRequest,
    SimilarityScoreUsingSpaceRequest
)


class PerceptionEncoderModelFrontend:
    def __init__(self):
        self.model_backend = PerceptionEncoderModelBackEnd()

    def get_model_id(self) -> str:
        return self.model_backend.model_id

    async def embed_text(self, embed_tex_request: EmbedTextRequest) -> EmbedTextResponse:
        return await self.model_backend.embed_text(embed_text_request=embed_tex_request)

    async def embed_image(self, embed_image_request: EmbedImageRequest) -> EmbedImageResponse:
        return await self.model_backend.embed_image(embed_image_request=embed_image_request)

    async def get_similarity_score(self,
                                   similarity_score_request: SimilarityScoreRequest) -> SimilarityScoreResponse:
        return await self.model_backend.get_similarity_score(similarity_score_request=similarity_score_request)

    async def get_similarity_score_using_space(self,
                                               request: SimilarityScoreUsingSpaceRequest) -> SimilarityScoreResponse:
        return await self.model_backend.get_similarity_score_using_space(request=request)