from eval.mpdocvqa import MPDocVQAEvaluator
from models.vlm.qwen_vl import Qwen2VL
from structure.vistree import VisRAGTree

if __name__ == '__main__':
    model = Qwen2VL("...")
    tree = VisRAGTree(model)
    prefix = "concat"
    evaluator = MPDocVQAEvaluator(tree, log=f"tmp/{prefix}.txt", strategy="concat")
    evaluator.load_dataset("lmms-lab/MP-DocVQA", split="val", oracle=False, cache_dir="data")

    evaluator.run()