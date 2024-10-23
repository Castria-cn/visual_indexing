from typing import List
from abc import abstractmethod
from models.meta import ImageLike

class TreeMeta:
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def build(self, images: ImageLike) -> None:
        pass

    @abstractmethod
    def query(self, prompt: str) -> str:
        pass