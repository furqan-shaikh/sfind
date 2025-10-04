import asyncio
import torch
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from PIL import Image

from sfind.models.models import (
    EmbedTextRequest, EmbedTextResponse,
    EmbedImageRequest, EmbedImageResponse, SimilarityScoreResponse, SimilarityScoreRequest,
    SimilarityScoreUsingSpaceRequest, ScoreRepresentation
)

class CLIPModelBackend:
    def __init__(self):
        self.model_id = "openai/clip-vit-base-patch32"
        self.model = AutoModel.from_pretrained(self.model_id,
                                          # dtype=torch.bfloat16,
                                          attn_implementation="sdpa")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def embed_text(self, embed_text_request: EmbedTextRequest) -> EmbedTextResponse:
        """Generate embedding for text."""
        # text_inputs = self.processor(text=request.text, return_tensors="pt").to(self.device)
        text_inputs = self.tokenizer(text=embed_text_request.text, return_tensors="pt").to(self.device)

        def _infer():
            with torch.no_grad():
                # outputs is a PyTorch tensor of shape [1, hidden_dim] (batch size = 1). Example shape: [1, 512]
                outputs = self.model.get_text_features(**text_inputs)
            emb = outputs / outputs.norm(dim=-1, keepdim=True)
            # embeddings = outputs[0].tolist()
            # embeddings = emb.squeeze(0).cpu().tolist()
            return EmbedTextResponse(embeddings=emb)

        return await asyncio.to_thread(_infer)

    async def embed_image(self, embed_image_request: EmbedImageRequest) -> EmbedImageResponse:

        def _infer():
            image = Image.open(embed_image_request.image_path)
            image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            # image_inputs = self.tokenizer(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**image_inputs)
            emb = outputs / outputs.norm(dim=-1, keepdim=True)
            # embeddings = outputs[0].tolist()
            # embeddings = emb.squeeze(0).cpu().tolist()
            return EmbedImageResponse(embeddings=emb)

        return await asyncio.to_thread(_infer)

    async def get_similarity_score(self,
                                   similarity_score_request: SimilarityScoreRequest) -> SimilarityScoreResponse:

        # def _compute():
        #
        #     # image = Image.open(io.BytesIO(embed_image_request.image))  # <-- convert bytes to PIL.Image
        #     image = Image.open(embed_image_request.image_path)
        #     labels = embed_text_request.text
        #
        #     inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        #     outputs = model(**inputs)
        #     logits_per_image = outputs.logits_per_image
        #     similarity_score = logits_per_image[0][0].item()
        #
        #     return SimilarityScoreResponse(score=similarity_score)
        #
        # return await asyncio.to_thread(_compute)

        def _compute(text_vector, image_vector):
            # 3. Cosine similarity
            # dot = np.dot(text_vector, image_vector)
            # norm_product = np.linalg.norm(text_vector) * np.linalg.norm(image_vector)
            # similarity = float(dot / norm_product) if norm_product > 0 else 0.0
            similarity = (similarity_score_request.text_embedding.to(torch.float32) @ similarity_score_request.image_embedding.T).item()
            return SimilarityScoreResponse(score=similarity)

        # # 1. Embed Text (async)
        # text_embedding_response = await self.embed_text(embed_text_request)
        # # 2. Embed image (async)
        # image_embedding_response = await self.embed_image(embed_image_request)

        # return await asyncio.to_thread(_compute,
        #                                text_vector=np.array(similarity_score_request.text_embedding, dtype=np.float32),
        #                                image_vector=np.array(similarity_score_request.image_embedding, dtype=np.float32))
        return await asyncio.to_thread(_compute,
                                       text_vector=similarity_score_request.text_embedding,
                                       image_vector=similarity_score_request.image_embedding)

    async def get_similarity_score_using_space(self, request: SimilarityScoreUsingSpaceRequest) -> SimilarityScoreResponse:

        def _compute():
            image = Image.open(request.file_path)
            labels = [request.text, ""]
            inputs = self.processor(text=labels, images=image, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            scores = [[ScoreRepresentation(score=value, score_display=f"{value:.10f}") for value in sublist] for sublist in probs.tolist()]
            return SimilarityScoreResponse(score=scores[0])

        return await asyncio.to_thread(_compute)
