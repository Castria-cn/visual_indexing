import pickle
from tqdm import tqdm
import numpy as np
import heapq
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
from models.base import VLMBase
from typing import List, Union
from models.base import ImageLike
from structure.tree import TreeBase
from utils.image_utils import zip_images

class VisRAGTree(TreeBase):
    def __init__(self,
                 model: VLMBase,
                 embed_model_path: str="openbmb/VisRAG-Ret",
                 block_size: int=4,
                 max_height: int=3,
                 root_nodes: int=3,
                 **kwargs):
        """
        - model: VLM which has a `predict` method.
        - embed_model_path: VisRAG embedding model path
        - block_size: block size when building the tree
        - max_height: max height of the tree
        - root_nodes: when top layer of the tree <= `root_nodes`, stop building the tree
        """
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model_path, trust_remote_code=True)
        self.embed_model = AutoModel.from_pretrained(embed_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0")
        self.embed_model.eval()
        self.block_size = block_size
        self.max_height = max_height
        self.root_nodes = root_nodes
        self.kwargs = kwargs

    def weighted_mean_pooling(self, hidden, attention_mask) -> torch.Tensor:
        attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
        d = attention_mask_.sum(dim=1, keepdim=True).float()
        reps = s / d
        return reps

    @torch.no_grad()
    def encode(self, text_or_image_list) -> np.ndarray:
        if (isinstance(text_or_image_list[0], str)):
            inputs = {
                "text": text_or_image_list,
                'image': [None] * len(text_or_image_list),
                'tokenizer': self.tokenizer
            }
        else:
            inputs = {
                "text": [''] * len(text_or_image_list),
                'image': text_or_image_list,
                'tokenizer': self.tokenizer
            }
        outputs = self.embed_model(**inputs)
        attention_mask = outputs.attention_mask
        hidden = outputs.last_hidden_state

        reps = self.weighted_mean_pooling(hidden, attention_mask)   
        embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
        return embeddings

    def get_scores(self, images: Union[np.ndarray, List[Image.Image]], query: str, to_numpy: bool=True) -> Union[np.ndarray, List[float]]:
        """
        Calculate similarity scores between `images` and `query`.
        - to_numpy: when set to `True`, return `numpy.ndarray`.
        """
        queries = [query]
        embeddings_query = self.encode(queries)
        if type(images) != np.ndarray:
            embeddings_doc = self.encode(images)
        else:
            embeddings_doc = images
        scores = (embeddings_query @ embeddings_doc.T).flatten()
        
        if not to_numpy:
            scores = scores.tolist()
        
        return scores
    
    def greedy_search(self, tree: List[List[Image.Image]], query: str, k: int=3) -> List[int]:
        top_items = self.get_scores(tree[-1], query, to_numpy=False)
        results = []
        pqueue = []
        for i, score in enumerate(top_items):
            heapq.heappush(pqueue, (-score, len(tree) - 1, i))
        
        while len(results) < k:
            _, layer, idx = heapq.heappop(pqueue)

            if layer == 0:
                results.append(idx)
                continue

            next_nodes = self.get_scores(tree[layer - 1][idx * self.block_size:(idx + 1) * self.block_size], query, to_numpy=False)

            for i, score in enumerate(next_nodes):
                heapq.heappush(pqueue, (-score, layer - 1, idx * self.block_size + i))
        
        return results
    
    def build(self, images: ImageLike, **kwargs) -> None:
        """
        11.5: pure image `build`.
        """
        layers = list()
        embed_layers = list()
        leaves = list()

        for image in images:
            leaves.append(image)
        
        layers.append(leaves)

        if kwargs['strategy'] == "retrieve":
            while len(layers) < self.max_height and len(layers[-1]) > self.root_nodes:
                new_layer = zip_images(layers[-1], block_size=self.block_size, pad=True)
                layers.append(new_layer)
        
        for layer in layers:
            embed_layers.append(self.encode(layer))
        
        self.tree = layers
        self.embed_tree = embed_layers

        if 'save_path' in kwargs:
            with open(kwargs['save_path'], 'wb') as fp:
                pickle.dump(self.embed_tree, fp)
    
    def retrieve(self, images: ImageLike, prompt: str, k: int=3) -> List[Image.Image]:
        if k >= len(images):
            return images
        if hasattr(self, 'embed_tree'):
            indices = self.greedy_search(self.embed_tree, prompt, k=k)
        else:
            indices = self.greedy_search(self.tree, prompt, k=k)

        return [images[idx] for idx in indices]
    
    def load(self, pickle_file: str) -> None:
        with open(pickle_file, 'rb') as fp:
            self.embed_tree = pickle.load(fp)