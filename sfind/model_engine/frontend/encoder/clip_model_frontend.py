from sfind.model_engine.backend.encoder.clip_model_backend import CLIPModelBackend
from sfind.models.models import EmbedTextRequest, EmbedTextResponse, EmbedImageRequest, EmbedImageResponse, \
    SimilarityScoreRequest, SimilarityScoreResponse, SimilarityScoreUsingSpaceRequest


class CLIPModelFrontend:
    def __init__(self):
        self.model_backend = CLIPModelBackend()

    def get_model_id(self) -> str:
        return self.model_backend.model_id

    async def embed_text(self, embed_text_request: EmbedTextRequest) -> EmbedTextResponse:
        return await self.model_backend.embed_text(embed_text_request=embed_text_request)

    async def embed_image(self, embed_image_request: EmbedImageRequest) -> EmbedImageResponse:
        return await self.model_backend.embed_image(embed_image_request=embed_image_request)

    async def get_similarity_score(self,
                                   similarity_score_request: SimilarityScoreRequest) -> SimilarityScoreResponse:
        return await self.model_backend.get_similarity_score(similarity_score_request=similarity_score_request)

    async def get_similarity_score_using_space(self,
                                               request: SimilarityScoreUsingSpaceRequest) -> SimilarityScoreResponse:
        return await self.model_backend.get_similarity_score_using_space(request=request)