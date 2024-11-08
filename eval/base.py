import time
import inspect
from typing import List, Any, Dict, Union
from abc import abstractmethod
from tqdm import tqdm
import pickle
from models.base import ImageLike

class UnpredictableError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class VLEvaluator:
    def __init__(self, log: str="tmp/eval_log.txt"):
        self.log = log
        with open(log, "w") as fp:
            fp.close()
    
    def log_str(self, text: str):
        text = str(text)
        with open(self.log, "a") as fp:
            fp.write(text)
            fp.close()

    @abstractmethod
    def load_dataset(self, **kwargs):
        """
        Set `self.dataset` to the following format:
        List({
            "qid": int(optional),
            "corpus": Any,
            "query": str,
            "answer": str | List[str],
            *other_keys, ...
        })

        Or, for overhead consideration `self.dataset` can be a generator, each time yield a dict.
        """
        pass
    
    @abstractmethod
    def predict(self, images: ImageLike, query: str, **kwargs) -> Dict[str, Any]:
        """
        Make prediction. When unable to predict, call `call_unpredictable`.
        """
        pass
    
    @abstractmethod
    def metric(self, predict: str, answer: Union[str, List[str]]) -> Any:
        pass
    
    def call_unpredictable(self, message: str):
        raise UnpredictableError(message)
    
    def build(self, images: ImageLike, **kwargs) -> None:
        pass

    def filter(self, item: Dict[str, Any]) -> bool:
        return False
    
    def run(self, samples: int=0, output_file: str="results.pkl") -> List[Dict]:
        """
        - samples: #samples to be evaluated.
        Each output dict has format: {
            "success": bool,
            "qid": int,
            "query": str,
            "time": float,
            **prediction_dict,
            (Optional)"exception": str,
            (Optional)"gt": str
        }
        """
        assert hasattr(self, 'dataset'), "Must call `load_dataset` first!"

        if inspect.isgeneratorfunction(self.dataset):
            dataset = self.dataset()
            print(dataset)
        else:
            dataset = self.dataset

        results = list()
        id_ = 0
        for item in tqdm(iter(dataset)):
            if samples != 0 and id_ >= samples:
                break
            if self.filter(item):
                continue
            try:
                images, query, answer = item.pop("corpus"), item.pop("query"), item.pop("answer")
                qid = id_ if "qid" not in item else item["qid"]
                id_ += 1

                self.build(images, **item) # build index
                output_dict = {
                    "qid": qid,
                    "query": query,
                    "gt": answer
                }

                start_time = time.perf_counter()

                result = self.predict(images, query, **item) # prediction

                output_dict.update(result)
                output_dict["success"] = True

                end_time = time.perf_counter()
                output_dict.update({"time": end_time - start_time})

                metric = self.metric(result, answer, **item) # calculate metrics
                output_dict["metric"] = metric

                results.append(output_dict)
            except Exception as e:
                print(e)
                results.append({
                    "success": False,
                    "exception": str(e)
                })
            self.log_str(str(results[-1]) + "\n")
            if len(results) % 20 == 0:
                with open(output_file, "wb") as f:
                    pickle.dump(results, f)
                    f.close()
                    
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
            f.close()
            
        return results