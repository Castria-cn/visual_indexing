from typing import List
from abc import abstractmethod
from models.vlm import ImageLike

class TreeMeta:
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def build(self, images: ImageLike) -> None:
        pass
    
    @abstractmethod
    def retrieve(self, prompt: str) -> List[int]:
        pass

    @abstractmethod
    def query(self, prompt: str) -> str:
        pass