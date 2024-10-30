import os
import datasets
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from typing import Any, List, Union
from eval.base import VLEvaluator
from models.base import ImageLike
from utils.pdf_utils import PDFReader
from structure.garlic import GarlicTree

class NarrativeQAEvaluator(VLEvaluator):
    def __init__(self, garlic: GarlicTree, log: str="tmp/eval_log.txt"):
        super().__init__(log)
        self.tree = garlic
        self.rouge = Rouge()

    def load_dataset(self, dataset_path: str, split: str="test"):
        self.reader = PDFReader()
        dataset = datasets.load_dataset(dataset_path)
        def generator():
            for item in dataset[split]:
                # print(item.keys())
                yield {
                    "corpus": item["document"]["summary"]["text"],
                    "query": item["question"]["text"],
                    "answer": [answer["text"] for answer in item["answers"]],
                    "answer_tokens": [answer["tokens"] for answer in item["answers"]],
                    "doc_id": item["document"]["id"],
                }
        
        self.dataset = generator
    
    def predict(self, corpus: str, query: str, **kwargs) -> str:
        if os.path.exists("tmp/" + kwargs["doc_id"] + "_tree.pkl"):
            self.tree.load("tmp/" + kwargs["doc_id"] + "_tree.pkl")
        else:
            self.tree.build_text(corpus, path="tmp/" + kwargs["doc_id"])
        return self.tree.query(query, export_traj="tmp/" + kwargs["doc_id"] + ".json")
        

    def metric(self, predict: str, answer: Union[str, List[str]], **kwargs) -> Any:
        rouges = [self.rouge.get_scores(predict, ans) for ans in answer]
        rouge_score = max([res[0]['rouge-l']['p'] for res in rouges])
        splitted_predict = nltk.word_tokenize(predict)
        bleu_score = sentence_bleu(kwargs["answer_tokens"], splitted_predict)
        return (rouge_score, bleu_score)