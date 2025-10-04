from typing import Protocol
from sfind.models.models import FetchRequest, FetchResponse, StoreRequest, Resource

class Storage(Protocol):
    """
    Abstract interface for pluggable embedding storage backends.

    A Storage implementation is responsible for persisting, retrieving, and
    enumerating embeddings associated with files or objects. Implementations
    may target local filesystems (via extended attributes), object stores
    (e.g. S3), or other backends.

    All methods are asynchronous to support both local and remote I/O.
    """

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

    async def set(self, store_request: StoreRequest) -> bool:
        """
        Store an embedding for the given resource.

        Args:
            store_request: Request containing resource identifier, model id, and embedding vector.
        Returns:
            True if the embedding was successfully stored, False otherwise.
        """

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
            root_path: Path or prefix within the storage system (e.g., local directory or S3 prefix).
            file_types: filter to apply for the file to be returned
        Returns:
            A list of Resource objects representing available files/objects.
        """
