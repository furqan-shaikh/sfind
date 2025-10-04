import asyncio
import torch
from PIL import Image

from perception_models.core.vision_encoder import pe, transforms
from sfind.models.models import (
    EmbedTextRequest, EmbedTextResponse,
    EmbedImageRequest, EmbedImageResponse,
    SimilarityScoreResponse, SimilarityScoreRequest, SimilarityScoreUsingSpaceRequest
)
from sfind.model_engine.utils import get_device
from sfind.utils.embeddings import list_to_tensor


class PerceptionEncoderModelBackEnd:
    def __init__(self):
        self.model_id = "PE-Core-L14-336"
        self.device = get_device()
        self.model = pe.CLIP.from_config(self.model_id, pretrained=True)  # Downloads from HF
        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)

    async def embed_text(self, embed_text_request: EmbedTextRequest) -> EmbedTextResponse:
        def _infer():
            tokenized_text = self.tokenizer(embed_text_request.text).to(self.device)
            with torch.inference_mode():
                _, text_features, _ = self.model(None, tokenized_text)
            return EmbedTextResponse(embeddings=text_features)

        return await asyncio.to_thread(_infer)


    async def embed_image(self, embed_image_request: EmbedImageRequest) -> EmbedImageResponse:
        def _infer():
            image = self.preprocess(Image.open(embed_image_request.image_path)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                image_features, _, _ = self.model(image, None)
            return EmbedImageResponse(embeddings=image_features)

        return await asyncio.to_thread(_infer)

    async def get_similarity_score(self,
                                   similarity_score_request: SimilarityScoreRequest) -> SimilarityScoreResponse:
        def _compute():
            # text_tensor = list_to_tensor(similarity_score_request.text_embedding)
            # features = text_tensor.to(torch.float32)
            # image_tensor = list_to_tensor(similarity_score_request.image_embedding)
            # text_probs = features @ image_tensor.T
            # return SimilarityScoreResponse(
            #     score=text_probs.squeeze(0).cpu().tolist()
            # )
            # print(text_probs)
            similarity = (similarity_score_request.text_embedding.to(
                torch.float32) @ similarity_score_request.image_embedding.T).item()
            return SimilarityScoreResponse(score=similarity)

        return await asyncio.to_thread(_compute)

    async def get_similarity_score_using_space(self, request: SimilarityScoreUsingSpaceRequest) -> SimilarityScoreResponse: ...