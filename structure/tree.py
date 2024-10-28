from typing import Any
from abc import abstractmethod
from models.base import ImageLike

class TreeBase:
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def build(self, images: ImageLike, **kwargs) -> None:
        pass

    @abstractmethod
    def query(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def load(self, archive: Any) -> None:
        pass