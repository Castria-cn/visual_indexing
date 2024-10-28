import inspect
from typing import List, Any, Dict, Union
from abc import abstractmethod
from tqdm import tqdm
from models.base import ImageLike

class UnpredictableError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class VLEvaluator:
    @abstractmethod
    def load_dataset(self, **kwargs):
        """
        Set `self.dataset` to the following format:
        List({
            "qid": int(optional),
            "images": ImageLike,
            "query": str,
            "answer": str | List[str],
            *other_keys, ...
        })

        Or, for overhead consideration `self.dataset` can be a generator, each time yield a dict.
        """
        pass
    
    @abstractmethod
    def predict(self, images: ImageLike, query: str, **kwargs) -> str:
        """
        Make prediction. When unable to predict, call `call_unpredictable`.
        """
        pass
    
    @abstractmethod
    def metric(self, predict: str, answer: Union[str, List[str]]) -> Any:
        pass
    
    def call_unpredictable(self, message: str):
        raise UnpredictableError(message)
    
    
    def run(self, samples: int=0, extra_keys: List[str]=None) -> List[Dict]:
        """
        - samples: #samples to be evaluated
        - extra_keys: extra keys to be passed from dataset item to predict.
        """
        assert hasattr(self, 'dataset'), "Must call `load_dataset` first!"

        if inspect.isgeneratorfunction(self.dataset):
            dataset = self.dataset()
            print(dataset)
        else:
            dataset = self.dataset

        result = list()
        id_ = 0
        for item in tqdm(iter(dataset)):
            if samples != 0 and id_ >= samples:
                break
            try:
                images, query, answer = item["images"], item["query"], item["answer"]
                qid = id_ if "qid" not in item else item["qid"]
                id_ += 1
                extra_dict = {}
                if extra_keys:
                    extra_dict = {item[key] for key in extra_keys}
                result = self.predict(images, query, **extra_dict)
                metric = self.metric(result, answer)
                result.append({
                    "success": True,
                    "qid": qid,
                    "query": query,
                    "answer": answer,
                    "metric": metric
                })
            except UnpredictableError as e:
                print(e)
                result.append({
                    "success": False
                })
            
        return result