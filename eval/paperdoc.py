import os
from typing import Any, List, Union
import jsonlines
from eval.base import VLEvaluator
from models.base import ImageLike
from utils.pdf_utils import PDFReader
from structure.garlic import GarlicTree

class PaperDocEvaluator(VLEvaluator):
    def __init__(self, garlic: GarlicTree, doc_dir: str, cache_dir: str, image_scale: float=0.75):
        self.doc_dir = doc_dir
        self.cache_dir = cache_dir
        self.tree = garlic
        self.image_scale = image_scale

    def load_dataset(self, jsonline_file: str):
        self.reader = PDFReader()
        def generator():
            with jsonlines.open(jsonline_file) as reader:
                for obj in reader:
                    pdf_path = self.doc_dir + "/" + obj["PDF name"] + ".pdf"
                    if not os.path.exists(pdf_path):
                        continue
                    self.reader.load(pdf_path)
                    images = list()
                    for i in range(len(self.reader)):
                        page = self.reader.capture(i, path=self.cache_dir + f"/page{i}.png", return_image=True)
                        w, h = page.size
                        page = page.resize((int(w * self.image_scale), int(h * self.image_scale)))
                        images.append(page)
                    print(f"PDF file with {len(images)} pages loaded.")
                    query = obj["Query"]
                    answer = obj["Answer"]
                    yield {
                        "images": images,
                        "query": query,
                        "answer": answer,
                        "pdf_name": obj["PDF name"]
                    }
        
        self.dataset = generator
    
    def predict(self, images: ImageLike, query: str, **kwargs) -> str:
        if os.path.exists("tmp/" + kwargs["pdf_name"] + "_tree.pkl"):
            self.tree.load("tmp/" + kwargs["pdf_name"] + "_tree.pkl")
        else:
            self.tree.build(images, path="tmp/" + kwargs["pdf_name"])
        return self.tree.query(query, export_traj="tmp/" + kwargs["pdf_name"] + ".json")
        

    def metric(self, predict: str, answer: Union[str, List[str]]) -> Any:
        return (predict, answer)