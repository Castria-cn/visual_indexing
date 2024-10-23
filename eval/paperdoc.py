import os
from typing import Any, List, Union
import jsonlines
from eval.base import VLEvaluator
from models.base import ImageLike
from utils.pdf_utils import PDFReader

class PaperDocEvaluator(VLEvaluator):
    def __init__(self, doc_dir: str, cache_dir: str):
        self.doc_dir = doc_dir
        self.cache_dir = cache_dir

    def load_dataset(self, jsonline_file: str):
        self.reader = PDFReader()
        def generator():
            with jsonlines.open(jsonline_file) as reader:
                for obj in reader:
                    pdf_path = self.doc_dir + "/" + obj["PDF name"] + ".pdf"
                    if not os.path.exists(pdf_path):
                        continue
                    self.reader.load(pdf_path)
                    images = [self.reader.capture(i, path=self.cache_dir + f"/page{i}.png") for i in range(len(self.reader))]
                    query = obj["Query"]
                    answer = obj["Answer"]
                    yield {
                        "images": images,
                        "query": query,
                        "answer": answer
                    }
        
        self.dataset = generator
    
    def predict(self, images: ImageLike, query: str) -> str:
        return ""

    def metric(self, predict: str, answer: Union[str, List[str]]) -> Any:
        return 0.0