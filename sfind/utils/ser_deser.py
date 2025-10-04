def str_to_bytes(s: str) -> bytes:
    """Convert a string to bytes using UTF-8 encoding."""
    return s.encode("utf-8")


def bytes_to_str(b: bytes) -> str:
    """Convert bytes back to a string using UTF-8 decoding."""
    return b.decode("utf-8")
