from dataclasses import dataclass
from typing import Union, List, Optional
import torch

IMAGE_FILE_TYPE = "i" or "I" or "image"

TYPE_REGISTER = {
    IMAGE_FILE_TYPE: [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
}

@dataclass
class ModelConfig:
    encoder_model_type: str
    captioning_model_type: str

@dataclass
class Config:
    model_config: ModelConfig

@dataclass
class Context:
    config: Config

@dataclass
class ScoreRepresentation:
    score: float
    score_display: str

@dataclass
class EmbedTextRequest:
    text: List[str]


@dataclass
class EmbedTextResponse:
    embeddings: torch.Tensor
    is_success: bool = True
    error_message: str = ""

@dataclass
class EmbedImageRequest:
    image: bytes = None
    name: str = ""
    image_path: str = ""

@dataclass
class EmbedImageResponse:
    embeddings: torch.Tensor
    is_success: bool = True
    error_message: str = ""

@dataclass
class SimilarityScoreResponse:
    score: Union[float | ScoreRepresentation]
    is_success: bool = True
    error_message: str = ""

@dataclass
class SimilarityScoreRequest:
    text_embedding: Union[list[float], torch.Tensor]
    image_embedding: Union[list[float], torch.Tensor]


@dataclass
class SimilarityScoreUsingSpaceRequest:
    text: List[str]
    file_path: str

@dataclass
class FetchCaptionRequest:
    file_data: Union[str | bytes]

@dataclass
class FetchCaptionResponse:
    caption: str
    is_success: bool = True
    error_message: str = ""

@dataclass
class RetrieveRequest:
    prompt: str
    path: str
    file_types: list[str]
    explain: bool = False
    limit: int = 10

@dataclass
class RetrieveResponse:
    similarity_score: ScoreRepresentation
    file_uri: str
    description: str = ""

@dataclass
class FetchRequest:
    file_path: str
    model_id: str
    type: str

@dataclass
class FetchResponse:
    data: bytes = None
    is_success: bool = True
    error: str = ""

@dataclass
class StoreRequest:
    file_path: str
    model_id: str
    data: bytes
    type: str

@dataclass
class StoreResponse:
    is_success: bool = True
    error: str = ""

@dataclass
class Resource:
    uri: str                      # "file:///home/x.png" or "oci://bucket/x.png"
    storage_type: str             # "local", "s3", "oci", etc.
    size: Optional[int] = None    # bytes
    last_modified: Optional[str] = None