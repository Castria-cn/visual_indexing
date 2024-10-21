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
    def attentioned_predict(self, prompt: str, output_only: bool=True, inner_attention: bool=False) -> Tuple[str, torch.Tensor]:
        pass

    @abstractmethod
    def align_token_str(self, splited: List[str], tokens: torch.Tensor) -> List[Tuple[int, int]]:
        """
        For each chunk in `splited`, find the [start, end) of it in `tokens`.
        NOTE: tokenize("".join(splited)) == tokens
        Outputs:
        List[Tuple[int, int]]: each element a [start, end) tuple.
        
        """
        pass
    
    @abstractmethod
    def str_level_score(self, splited_input: List[str], splited_output: List[str], attentions: torch.Tensor, io_attention: bool=True) -> torch.Tensor:
        """
        - splited_input: The splited format of the input. e.g. ["What is the derivative of function", "f(x) = e^x?"]
        - splited_output: The splited format of the output. e.g. ["The derivative of function f(x) = e^x is ] NOTE: Not include input sequence.
        - attentions: tensor of shape [output_len, input_len].
        - io_attention: set to `True` when calculating input-output attention. When calculating input-input attention, set to `False`.
        NOTE: splited_input, splited_output must be same as input, output.
        """
        pass

class VLMMeta(ABC):
    model: PreTrainedModel
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, prompt: str, image: ImageLike) -> str:
        pass