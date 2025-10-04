import asyncio
from pathlib import Path

import xattr
from sfind.models.models import FetchRequest, FetchResponse, StoreRequest, Resource, StoreResponse

EMBEDDING_XATTR_KEY_TEMPLATE = "user.sfind.embedding.{}"
CAPTION_XATTR_KEY_TEMPLATE = "user.sfind.caption.{}"

class VFSFileSystem:
    def __init__(self):
        """

        :rtype: object
        """
        self.storage_type = "local"

    async def get(self, fetch_request: FetchRequest) -> FetchResponse:
        """
        Retrieve an embedding for the given resource.

        Args:
            fetch_request: Request containing file/resource identifier and model id.
        Returns:
            FetchResponse containing the embedding vector and related metadata.
        Raises:
            KeyError if the embedding does not exist.
        """

        def _read_xattr():
            attr = self._get_attr_key(model_id=fetch_request.model_id, type=fetch_request.type)
            try:
                response = xattr.getxattr(f=fetch_request.file_path,attr=attr)
                return FetchResponse(data=response)
            except Exception as e:
                error_message = f"Error getting extended attribute: {attr} : {e}"
                return FetchResponse(is_success=False, error=error_message)

        # inside an async function, any blocking call blocks the event loop
        return await asyncio.to_thread(_read_xattr)

    async def set(self, store_request: StoreRequest) -> StoreResponse:
        """
        Store an embedding for the given resource.

        Args:
            store_request: Request containing resource identifier, model id, and embedding vector.
        Returns:
            True if the embedding was successfully stored, False otherwise.
        """
        def _set_xattr():
            attr = self._get_attr_key(model_id=store_request.model_id, type=store_request.type)
            try:
                xattr.setxattr(f=store_request.file_path, attr=attr, value=store_request.data)
                return StoreResponse()
            except Exception as e:
                error_message = f"Error setting extended attribute: {attr} : {e}"
                return StoreResponse(is_success=False, error=error_message)

        # inside an async function, any blocking call blocks the event loop
        return await asyncio.to_thread(_set_xattr)

    async def has(self, fetch_request: FetchRequest) -> bool:
        """
        Check whether an embedding exists for the given resource.

        Args:
            fetch_request: Request containing resource identifier and model id.
        Returns:
            True if an embedding exists, False otherwise.
        """
    async def list_files(self, root_path: str, file_types: list[str]) -> list[Resource]:
        """
        Enumerate resources under the given root path or prefix.

        Args:
            root_path: Path or prefix within the storage system
            file_types: filter to apply for the file to be returned
        Returns:
            A list of Resource objects representing available files/objects.
        """
        base_path = Path(root_path)
        files: list[Resource] = []
        for ext in file_types:
            for file_path in base_path.rglob(f'*{ext}'):
                files.append(Resource(uri=file_path, storage_type=self.storage_type))

        return files

    def _get_attr_key(self, model_id: str, type: str) -> str:
        return EMBEDDING_XATTR_KEY_TEMPLATE.format(model_id) if type=="embedding" else CAPTION_XATTR_KEY_TEMPLATE.format(model_id)