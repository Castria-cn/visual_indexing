from Levenshtein import distance as levenshtein_distance
from typing import List, Union
from rouge import Rouge
import pickle

rouge = Rouge()
def calculate_anls(predict: str, ref: Union[str, List[str]], parse=True) -> float:
    """
    Calculate ANLS \in [0, 1] between `string1` and `string2`.
    """
    if type(ref) == str and ref.startswith('[') and parse:
        try: # some str starts with `[`, but not a list!
            ref = eval(ref)
        except:
            pass
    if type(ref) == str:
        lev_distance = levenshtein_distance(predict, ref)
        max_length = max(len(predict), len(ref))
        normalized_distance = lev_distance / max_length
        return 1 - normalized_distance

    return max(calculate_anls(predict, r, parse=False) for r in ref)

def calculate_rouge(predict: str, ref: Union[List[str], str]) -> float:
    """
    Calculate ROUGE \in [0, 1] between `string1` and `string2`.
    """
    if type(ref) == str and ref.startswith('['):
        ref = eval(ref)
    rouges = [rouge.get_scores(pre, ans) for ans in ref for pre in [predict, predict.lower()]]
    rouge_score = max([res[0]['rouge-l']['p'] for res in rouges])
    return rouge_score

name2func = {
    "anls": calculate_anls,
    "rouge": calculate_rouge
}

def report_score(file: str, metrics: List[str]=["anls", "rouge"], samples: int=None) -> None:
    assert all([metric in name2func for metric in metrics])
    
    if file.endswith(".pkl"):
        with open(file, "rb") as fp:
            obj = pickle.load(fp)
    else:
        obj = []
        with open(file, "r", encoding='utf-8') as fp:
            for line in fp:
                obj.append(eval(line.strip()))
    
    print(f'Result of file {file}:')

    for name in metrics:
        results = []
        metric = name2func[name]
        for item in obj[:samples]:
            if not item['success']:
                results.append(0.0)
                continue
            answer = item['answer']
            predict = item['predict']
            results.append(metric(predict, answer))

        print(f"{name}: {sum(results) / len(results)}")