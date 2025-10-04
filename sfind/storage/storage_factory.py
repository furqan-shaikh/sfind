from sfind.storage.interfaces.storage import Storage
from sfind.storage.local_filesystem.vfs_file_system import VFSFileSystem


def get_storage(path: str) -> Storage:
    """
    Factory that returns a Storage implementation depending on the path.

    Args:
        path: resource path (local filesystem or S3).

    Returns:
        Storage implementation.
    """
    if path.startswith("s3://"):
        # Extract bucket and prefix if needed
        raise Exception("S3 not yet supported")
    else:
        # Default: local filesystem
        return VFSFileSystem()
