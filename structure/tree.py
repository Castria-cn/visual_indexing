from typing import List
from abc import abstractmethod
from models.base import ImageLike, VLMBase

class TreeBase:
    model: VLMBase
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def build(self, images: ImageLike) -> None:
        pass
    
    @abstractmethod
    def retrieve(self, images: ImageLike, prompt: str) -> List[int]:
        pass

    @abstractmethod
    def query(self, prompt: str) -> str:
        pass

    @abstractmethod
    def load(self, file_path: str) -> None:
        pass