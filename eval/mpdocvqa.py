import os
import numpy as np
import datasets
import hashlib
from PIL import Image
from rouge import Rouge
from typing import Any, List, Union
from eval.base import VLEvaluator
from models.base import ImageLike
from utils.pdf_utils import PDFReader
from structure.vistree import VisRAGTree

class MPDocVQAEvaluator(VLEvaluator):
    def __init__(self,
                 tree: VisRAGTree,
                 log: str="tmp/eval_log.txt",
                 target_image_size: int=1152,
                 cache_dir: str="index",
                 strategy: str="concat",
                 **kwargs):
        """
        - tree: `VisRAGTree` object.
        - log: evaluation log path
        - target_image_size: max side length of the images
        - cache_dir: directory to save index files
        - strategy: must be in ['concat', 'visrag', 'retrieve']

        Available keyword arguments:
        - concat_size(default to 4)
        - visrag_k(default to 3)
        - retrieve_k(default to 3)
        """
        super().__init__(log)
        self.kwargs = kwargs
        self.tree = tree
        self.rouge = Rouge()
        self.cache_dir = cache_dir
        self.strategy = strategy
        self.target_size = target_image_size
    
    def merge_images_horizontally(self, image_list):
        widths, heights = list(), list()
        for img in image_list:
            widths.append(img.size[0])
            heights.append(img.size[1])
        
        total_width = sum(widths)
        max_height = max(heights)

        new_image = Image.new('RGB', (total_width, max_height))
        
        x_offset = 0
        for img in image_list:
            new_image.paste(img, (x_offset, 0))
            x_offset += img.width

        return new_image

    def adjust_size(self, img):
        w, h = img.size
        max_ratio = max(img.size[0] / self.target_size, img.size[1] / self.target_size)

        return img.resize((int(w / max_ratio), int(h / max_ratio)))

    def load_dataset(self, dataset_path: str, split: str="validation", oracle: bool=False, cache_dir=None):
        self.reader = PDFReader()
        dataset = datasets.load_dataset(dataset_path, cache_dir=cache_dir)
        def generator():
            for item in dataset[split]:
                images = [item[f'image_{i}'] for i in range(1, 21)]
                while images[-1] is None:
                    images.pop()
                resized_images = []
                for img in images:
                    resized_images.append(self.adjust_size(img))
                
                if oracle:
                    resized_images = [resized_images[int(item["answer_page_idx"])]]

                yield {
                    "corpus": resized_images,
                    "query": item["question"],
                    "answer": item["answers"],
                    "doc_id": item["doc_id"],
                    "page_ids": item["page_ids"]
                }
        
        self.dataset = generator
    
    def build(self, images: ImageLike, **kwargs) -> None:
        if self.strategy == "concat":
            return
        pages = kwargs['page_ids']
        hashed = hashlib.md5(pages.encode("utf-8")).hexdigest()
        pickle_dir = f'{self.cache_dir}/{hashed}.pkl'
        if os.path.exists(pickle_dir):
            self.tree.load(pickle_dir)
        else:
            self.tree.build(images, save_path=pickle_dir, strategy=self.strategy, **kwargs)
    
    def predict(self, images: ImageLike, query: str, **kwargs) -> str:
        strategy = self.strategy

        new_images = []
        if strategy == 'concat':
            concat_size = self.kwargs.get('concat_size', 4)
            if len(images) >= concat_size:
                for i in range(0, len(images), concat_size):
                    start = i
                    end = i + concat_size
                    cat = self.merge_images_horizontally(images[start:end])
                    new_images.append(cat)
            else:
                new_images = images

        elif strategy == 'retrieve':
            k = self.kwargs.get('retrieve_k', 3)
            new_images = self.tree.retrieve(images, query, k=k)

        elif strategy == 'visrag':
            k = self.kwargs.get('visrag_k', 3)
            scores = self.tree.get_scores(self.tree.embed_tree[0], query, to_numpy=True)
            indices = np.argsort(scores)[::-1][:k]
            new_images = [images[idx] for idx in indices]

        new_images = [self.adjust_size(img) for img in new_images]

        # predict by qwen2-vl
        examples = "Who is Miss Delmer?\nthe elderly spinster aunt of the Earl de Verseley and Captain Delmar.\nWho is the bully that steals Percival's lunch?\nhis teacher, Mr. O'Gallagher\nWhat news does Percival receive at the end of the story?\nHe has been granted the right to use his father's name, Delmar\n"
        prompt = f"Answer the question as concisely as you can. Here are some examples:\n{examples}\nQuestion: \n{query}"
        return self.tree.model.predict(prompt, new_images, return_attn=False)

    def metric(self, predict: str, answer: Union[str, List[str]], **kwargs) -> Any:
        if type(answer) == str:
            answer = eval(answer)
        rouges = [self.rouge.get_scores(pre, ans) for ans in answer for pre in [predict, predict.lower()]]
        rouge_score = max([res[0]['rouge-l']['p'] for res in rouges])
        return rouge_score