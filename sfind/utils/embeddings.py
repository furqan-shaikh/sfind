import io
from typing import List, Union

import numpy as np
import torch
from pyarrow import Tensor


def serialize_embedding(embedding: Union[list[float] | Tensor]) -> bytes:
    if isinstance(embedding, torch.Tensor):
        buffer = io.BytesIO()
        torch.save(embedding, buffer)
        return buffer.getvalue()
    else:
        arr = np.array(embedding, dtype=np.float32)  # choose float32
        return arr.tobytes()

def deserialize_embedding(data: bytes, type: str) -> Union[list[float] | Tensor]:
    #
    if type == "list":
        arr = np.frombuffer(data, dtype=np.float32)
        return arr.tolist()
    elif type == "tensor":
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location=None, weights_only=True)
    else:
        raise ValueError("return_type must be 'list' or 'tensor'")


def list_to_tensor(embedding: List[float], batch: bool = False) -> torch.Tensor:
    """
    Convert a Python list of floats to a PyTorch tensor.

    Args:
        embedding: list of floats
        batch: if True, adds a batch dimension -> shape [1, N]

    Returns:
        torch.Tensor
    """
    tensor = torch.tensor(embedding, dtype=torch.float32)
    if batch:
        tensor = tensor.unsqueeze(0)
    return tensor


def tensor_to_list(tensor: torch.Tensor) -> List[float]:
    """
    Convert a PyTorch tensor back to a list of floats.

    Args:
        tensor: 1D or 2D tensor
    Returns:
        list of floats
    """
    # If 2D with batch, remove batch dimension
    if tensor.dim() == 2 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    return tensor.cpu().tolist()
