import torch
from abc import ABC, abstractmethod
from typing import List, Union, Tuple
from PIL import Image
from transformers import PreTrainedModel

ImageLike = Union[str, Image.Image, List[Union[str, Image.Image]]]

def preprocess(image: ImageLike) -> List[Image.Image]:
    if not isinstance(image, list):
        image = [image]
    image = [(Image.open(img) if isinstance(img, str) else img).convert("RGB") for img in image]
    return image

class LLM(ABC):
    model: PreTrainedModel
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, prompt: str, return_attn: bool=False) -> Union[str, Tuple[str, Tuple]]:
        """
        - return_attn: when set to `True`, return an additional Tuple(Tensor[bs, num_heads, seq_len, seq_len], ...)
        """
        pass

    @abstractmethod
    def attentioned_predict(self, chunks: List[str]) -> Union[str, Tuple[str, Tuple]]:
        """
        - chunks: List of length `seq_len`, each element a str.
        Outputs:
        A matrix of shape [output_]
        """
        pass

class VLMMeta(ABC):
    model: PreTrainedModel
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, prompt: str, image: ImageLike) -> str:
        pass