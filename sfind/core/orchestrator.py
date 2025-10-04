from sfind.config.config import Context
from sfind.model_engine.model_factory import ModelFactory
from sfind.models.models import RetrieveRequest, RetrieveResponse, EmbedTextRequest, StoreRequest, \
    SimilarityScoreRequest, FetchRequest, EmbedImageResponse, EmbedImageRequest, FetchCaptionRequest
from sfind.storage.interfaces.storage import Storage
from sfind.storage.storage_factory import get_storage
from sfind.utils.embeddings import serialize_embedding, deserialize_embedding
from sfind.utils.ser_deser import bytes_to_str, str_to_bytes


class Orchestrator:
    def __init__(self, context: Context):
        self.context = context
        # Get the model front end based on config.
        self.encoder_model_front_end = ModelFactory.get_encoder_model(context=self.context)
        self.captioning_model_front_end = None


    async def execute(self, request: RetrieveRequest) -> list[RetrieveResponse]:
        # 1. Based on the path scheme, get the storage type
        storage = get_storage(request.path)
        # 2. Apply similarity semantics using embeddings
        response = await self._retrieve(request=request, storage=storage)
        # 3. apply limits
        response = self._apply_limits(request, response)
        if request.explain is False:
            return response

        # 4. Apply captioning if user has asked for it
        self.captioning_model_front_end = ModelFactory.get_captioning_model(context=self.context)
        await self._do_captioning(request=request, responses=response, storage=storage)
        return response

    async def _retrieve(self, request: RetrieveRequest, storage: Storage) -> list[RetrieveResponse]:
        model_id = self.encoder_model_front_end.get_model_id()
        # 3. Get all the image paths
        files = await storage.list_files(root_path=request.path, file_types=request.file_types)
        # 4. Compute text embedding for the prompt
        text_embedding = await self.encoder_model_front_end.embed_text(EmbedTextRequest(text=[request.prompt]))

        # for each image path
        #   check if it already has the embedding
        #   if yes, call the similarity score
        #   if no, compute the embedding, store in storage
        #     call the similarity score method
        # sort the similarity scores and return
        scores = []
        for file in files:
            file_path = file.uri
            file_embedding_response, should_store = await self._get_file_embedding(storage=storage, file_path=file_path)
            if should_store is True:
                await storage.set(store_request=StoreRequest(
                    model_id=model_id,
                    file_path=file_path,
                    data=serialize_embedding(file_embedding_response.embeddings),
                    type="embedding"
                ))
            score_response = await self.encoder_model_front_end.get_similarity_score(similarity_score_request=SimilarityScoreRequest(
                text_embedding = text_embedding.embeddings,
                image_embedding=file_embedding_response.embeddings
            ))
            # score_response = await self.model_front_end.get_similarity_score_using_space(request=SimilarityScoreUsingSpaceRequest(
            #     text=[request.prompt],
            #     file_path=file_path
            # ))
            scores.append(RetrieveResponse(
                similarity_score=score_response.score,
                file_uri=str(file_path)
            ))
        sorted_scores = sorted(scores, key=lambda score_item: score_item.similarity_score, reverse=True)
        return sorted_scores

    async def _do_captioning(self, request: RetrieveRequest, storage: Storage,  responses: list[RetrieveResponse]):
        for item in responses:
            # 1. Get captioning from storage
            response = await storage.get(fetch_request=FetchRequest(
                file_path=item.file_uri,
                model_id=self.encoder_model_front_end.get_model_id(),
                type="captioning"
            ))
            # 2. If present, read it
            if response.is_success is True:
                item.description = bytes_to_str(response.data)
                continue
            # 3. If not present, do captioning
            captioning_response = await self.captioning_model_front_end.get_caption(caption_request=FetchCaptionRequest(
                file_data=item.file_uri
            ))
            if captioning_response.is_success:
                item.description = captioning_response.caption
                # 4. Store in storage
                await storage.set(store_request=StoreRequest(
                    model_id=self.encoder_model_front_end.get_model_id(),
                    file_path=item.file_uri,
                    data=str_to_bytes(captioning_response.caption),
                    type="captioning"
                ))


    async def _get_file_embedding(self, storage: Storage, file_path: str) -> (EmbedImageResponse, bool):
        response = await storage.get(fetch_request=FetchRequest(
            file_path=file_path,
            model_id=self.encoder_model_front_end.get_model_id(),
            type="embedding"
        ))
        if response.is_success is True:
            file_embedding_response = EmbedImageResponse(
                embeddings=deserialize_embedding(response.data, type="tensor"),
            )
            return file_embedding_response, False
        else:
            response = await self.encoder_model_front_end.embed_image(embed_image_request=EmbedImageRequest(image_path=file_path))
            return response, True

    def _apply_limits(self, request, response):
        # if items in response are less than limits return
        if len(response) <= request.limit:
            return response
        # apply limits to return top 'limit' items
        return response[:request.limit]