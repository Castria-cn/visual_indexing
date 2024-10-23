import json
import torch
import pickle
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Any, Set

from models.meta import VLMMeta, ImageLike, preprocess, LLMMeta
from structure.tree import TreeMeta

class GarlicNode:
    node_id = 0
    children: List[Any]

    def __init__(self, image: Union[ImageLike, None]=None, desc: Union[str, None]=None) -> None:
        self.image = image
        self.desc = desc
        self.children = []

        self.id = GarlicNode.node_id
        GarlicNode.node_id += 1
    
    @property
    def image_num(self) -> int: # BUG
        if len(self.children) == 0:
            return 1
        if not hasattr(self, 'image_num'):
            self._image_num = sum([child[0].image_num for child in self.children])

        return self._image_num
    
    def add_children(self, children: List[Any]) -> None:
        self.children.extend(children)

class GarlicTree(TreeMeta):
    model: VLMMeta
    def __init__(self, vmodel: VLMMeta, lmodel: Union[VLMMeta, LLMMeta], log: str=None, max_tries: int=10):
        self.model = vmodel
        self.lmodel = lmodel
        self.max_tries = max_tries
        self.prompt_template = "Please read and summarize the following paragraphs carefully, Split your summary into different summary points according to the semantic information in these information points. It is not necessary to generate each summary point for each information point. Gather and organize information into summary points. In each summary point, try to avoid using pronouns like he/she/they and instead use full names. Generate in the format of: * <summary point1>\n* <summary point2>\n* <summary point3>."
        self.interm_template = "Please group the summary points, for each group, give it a subtitle and briefly describe it. Do not repeat the information point in the given paragraphs. Do not list any specific figure or number(For example, percentage / number...). Generate in the format of: **subtitle 1**: <description of subtitle1>\n**subtitle 2**: <description of subtitle2>\n..."
        self.prompt_template_img = "Are there any figures(pie chart, table, bar chart, etc.) in this document? If so, briefly summarize meaning of each figure in this page. Else, reply a single string `No`. Do not list any specific number or value."
        if log:
            self.log = log
            with open(self.log, "w") as fp:
                fp.close()
    
    def log_str(self, text: Any, newline: int=200, tag=None) -> None:
        text = str(text)
        def insert_newline(s, n):
            return '\n'.join([s[i:i+n] for i in range(0, len(s), n)])
        with open(self.log, "a") as fp:
            if tag:
                fp.write(f"<{tag}>\n{insert_newline(text, newline)}\n</{tag}>\n")
            else:
                fp.write(insert_newline(text, newline))
    
    def as_json(self, file_path: str) -> None:
        obj = {
            "nodes": [],
            "edges": []
        }
        for i, layer in enumerate(self.layers):
            for node in layer:
                obj["nodes"].append({
                    "id": node.id,
                    "desc": node.desc,
                    "layer": i
                })
                for (child, weight) in node.children:
                    self.log_str(str(child))
                    obj["edges"].append([node.id, child.id, weight])
        
        with open(file_path, "w") as fp:
            json.dump(obj, fp)
    
    def _save_leaves(self, nodes: List[GarlicNode], file_path: str):
        with open(file_path, "wb") as fp:
            pickle.dump(nodes, fp)
    
    def _save_tree(self, file_path: str):
        with open(file_path, "wb") as fp:
            pickle.dump(self.layers, fp)

    def load(self, file_path: str):
        """
        Load the tree from .pkl file.
        """
        with open(file_path, "rb") as fp:
            self.layers = pickle.load(fp)
            GarlicNode.node_id = sum([len(layer) for layer in self.layers])
            # calculate adj matrix
            self.adj_matrix = torch.zeros((GarlicNode.node_id, GarlicNode.node_id), dtype=torch.float32)
            for layer in self.layers:
                for node in layer:
                    node: GarlicNode
                    ids, weights = self._fetch_weights(node)
                    if len(ids):
                        self.adj_matrix[node.id][ids] = weights
            # id2node mapping    
            self.id2node = dict()
            for layer in self.layers:
                for node in layer:
                    self.id2node[node.id] = node
            self.log_str(f"Tree loaded.\n")

    def _load_leaves(self, file_path: str) -> List[GarlicNode]:
        with open(file_path, "rb") as fp:
            leaves = pickle.load(fp)
            GarlicNode.node_id = len(leaves)
            self.log_str(f"{len(leaves)} leaves loaded.\n")
        return leaves

    def _fetch_images(self, nodes: List[GarlicNode]) -> ImageLike:
        return [node.image for node in nodes]
    
    def _fetch_desc(self, nodes: List[GarlicNode]) -> List[str]:
        return [node.desc for node in nodes]
    
    def _fetch_weights(self, node: GarlicNode, return_tensors: str="pt") -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        assert return_tensors in ["pt", "np"]
        if len(node.children) == 0:
            return [], []
        nodes, weights = zip(*node.children)
        ids = [node.id for node in nodes]
        if return_tensors == "pt":
            return torch.tensor(ids), torch.tensor(weights)
        elif return_tensors == "pt":
            return np.array(ids), np.array(weights)

        raise NotImplementedError()
    
    def merge_images(self, nodes: List[GarlicNode], layer: int) -> Image.Image:
        """
        Merge bottom nodes into a single new image.
        - layer: layer id (after this merge)
        """
        # if layer == 1:
        #     images = self._fetch_images(nodes)
        #     images = [np.asarray(img) for img in images]
        #     concated = np.concatenate(images, axis=1)
        #     merged = Image.fromarray(concated)
        #     w, h = merged.size
        #     merge = merged.resize((w // 2, h // 2))
        # TODO

        return None
    
    def merge_text(self, nodes: List[GarlicNode], layer: int) -> Tuple[List[str], torch.Tensor]:
        """
        Merge the bottom layer texts into new strings. 
        """
        texts = self._fetch_desc(nodes)
        texts = [ip.strip().strip('*').replace('<|im_end|>', '') for ip in texts]
        model_inputs = [
            self.lmodel.pre_prompts + self.get_prompt_template(layer),
            *texts,
            self.lmodel.post_prompts
        ]
        self.log_str("Raw inputs: \n" + str(model_inputs) + "\n")
        new_text, attentions = self.lmodel.attentioned_predict("".join(model_inputs))
        self.log_str(f"Raw prediction: {new_text}")
        IPs = self._parse_response(new_text)
        scores = self.lmodel.str_level_score(model_inputs, IPs, attentions) # [len(IPs), len(nodes) + 2]
        scores = scores[:, 1: -1] # remove chat template
        
        assert len(IPs) == scores.shape[0]
        assert len(nodes) == scores.shape[1]

        if torch.any(torch.isnan(scores)):
            self.log_str("!!!Raw inputs: \n" + "".join(model_inputs) + "\n")
            new_text, attentions = self.lmodel.attentioned_predict("".join(model_inputs))
            self.log_str(f"!!!Raw prediction: {new_text}")
            self.log_str(torch.any(torch.isnan(attentions)))
            self.log_str(f"attentions: {attentions}")
            self.log_str(f"scores: {scores}")

        return IPs, scores
    
    def get_prompt_template(self, layer: int):
        """
        template when doing summarization.
        """
        if layer == 0:
            return self.prompt_template
        return self.prompt_template
    
    def get_query_template(self, query: str, next_nodes: List[GarlicNode]):
        """
        template when doing expansion.
        """
        template = f"Given the question: {query} and the following descriptions of a document's different part, which part is most likely to contain the information to answer this question? Reply the description id with a single number. Do not provide explanation."
        for i, node in enumerate(next_nodes):
            template += f"**Description {i + 1}:** " + node.desc + "\n"
        
        return template
    
    def _parse_response(self, response: str, split_char: str="*") -> List[str]:
        """
        * IP1
        * IP2
        ...
        NOTE: "".join(result) MUST EQUALS `response`
        """
        self.log_str(f"Raw response: {response}")
        positions = [i for i, c in enumerate(response) if c == split_char]
        if len(positions) <= 1:
            return [response]
        result = list()
        for i in range(len(positions)):
            if i == 0:
                result.append(response[:positions[1]])
            else:
                result.append(response[positions[i]: positions[i + 1] if i < len(positions) - 1 else None])
        
        self.log_str(f"Parsed reponse: {result}")

        assert "".join(result) == response

        return result

    def summary(self, nodes: List[GarlicNode], layer: int) -> List[GarlicNode]:
        """
        - layer: id of this layer(the summary result layer). 0 for bottom layer.
        """
        descs, scores = self.merge_text(nodes, layer)
        new_image = [None for i in range(len(descs))] # self.merge_images(nodes, layer)
        smry = "<NEXT_NODE>\n".join(descs)
        self.log_str(f"****************************************************************************\nSummary:\n {smry}\n****************************************************************************\n")
        new_nodes = [GarlicNode(new_image, desc) for desc in descs]
        # TODO add_children and weight in GARLIC
        for i in range(len(new_nodes)): 
            for j in range(len(nodes)):
                new_nodes[i].add_children([(nodes[j], scores[i, j].item())])
                assert new_nodes[i].id != nodes[j].id
        return new_nodes
    
    def select_page_num(self, nodes: List[GarlicNode], idx: int, token_sum: int=2048) -> int:
        approx_token_lens = [len(node.desc) for node in nodes]
        token_len, page_num = approx_token_lens[idx], 1
        while token_len < token_sum and idx + page_num < len(nodes):
            token_len += approx_token_lens[idx + page_num]
            page_num += 1
        page_num = max(2, page_num)
        self.log_str(f"Select {page_num} pages, approx token = {token_len}\n")
        
        return page_num
    
    def build(self, images: ImageLike, leaves: str=None) -> None:
        """
        The basic function of Garlic Tree. Begin from a list of images, garlic tree do:
        1. For each image I, summarize its text content T(I), then form a leaf node (I, T(I)). All of the leaves form the bottom layer L0.
        2. Do while P():
            - Summarize neighbor children to form a new summarization S. The attn score cannot be calculated (because of the flash-attn stuff)
        """
        self.layers = list()
        self.image_descs = list()
        leaves = list() if leaves is None else self._load_leaves(leaves)

        if len(leaves) == 0:
            for i, image in enumerate(images):
                w, h = image.size
                self.log_str(f"Predicting page {i}...\n")
                response = self.model.predict(self.get_prompt_template(0), image)
                IPs = self._parse_response(response)
                leaves.extend([GarlicNode(image=None, desc=ip) for ip in IPs])
                # leaves.append(GarlicNode(image=image, desc=self.model.predict(self.get_prompt_template(0), image))) # Original
                self.image_descs.append(GarlicNode(image=image, desc=self.model.predict(self.prompt_template_img, image)))
                self.log_str(f"Summarize of page {i}: \n***************************************************************************************************\n{leaves[-1].desc}\n")
                self.log_str(f"Summarize of figure: \n{self.image_descs[-1].desc}\n***************************************************************************************************\n")

            self._save_leaves(leaves, "leaves.pkl")
        self.layers.append(leaves)

        while len(self.layers) < 3 and len(self.layers[-1]) > 3:
            idx = 0
            new_layer = list()

            while idx < len(self.layers[-1]):
                page_num = self.select_page_num(self.layers[-1], idx) # key function 1: select chunk size
                nodes = self.layers[-1][idx: idx+page_num]

                new_nodes = self.summary(nodes, layer=len(self.layers)) # key function 2: summary

                new_layer.extend(new_nodes)
                idx += page_num

            self.layers.append(new_layer)
        
        self.layers[-1].extend(self.image_descs)

        self.id2node = dict()
        for layer in self.layers:
            for node in layer:
                self.id2node[node.id] = node
        
        # calculate adj matrix
        self.adj_matrix = torch.zeros((GarlicNode.node_id, GarlicNode.node_id), dtype=torch.float32)
        for layer in self.layers:
            for node in layer:
                node: GarlicNode
                ids, weights = self._fetch_weights(node)
                if len(ids):
                    self.adj_matrix[node.id][ids] = weights
        
        self.log_str(str(self.adj_matrix))

        self._save_tree("tree.pkl")
        
    def finish_query(self, prompt: str, memory: List[GarlicNode], images: ImageLike=None, multi_models: bool=False) -> Tuple[str, torch.Tensor]:
        """
        Given prompt and memory, ask LLM(VLM) if the question can be answered.
        """
        template = [self.lmodel.pre_prompts + "Can this question be answered by the following information? Response \"Yes\" or \"No\" in one word. Do not provide any explanation.\n\n",
                    f"Question: {prompt}\n\n",
                    f"Information:\n",
                    *[node.desc.replace('<|im_end|>', '') for node in memory],
                    self.lmodel.post_prompts
                    ]
        if multi_models and images is None:
            self.log_str(f"Raw finish input: {template}\n")
            output, attentions = self.lmodel.attentioned_predict("".join(template), inner_attention=True, max_new_tokens=100)
            scores = self.lmodel.str_level_score(template, template, attentions, io_attention=False) # [3 + len(memory), 3 + len(memory)]
            self.log_str(str(scores), tag="Score before adjusting")
            scores = scores[3: -1, 1] # [len(memory)]
            # compensation on position
            scores = scores * torch.arange(1, len(memory) + 1).to(scores)
            return output, scores
        return self.model.predict(template, images), torch.ones(len(memory) + 1)

    def graph_search(self, nodes: List[GarlicNode], memory_ids: Set[int], attentions: torch.Tensor) -> GarlicNode:
        """
        Search the next node to be queried.
        """
        # construct query-node similarity vector
        similarity = torch.zeros(GarlicNode.node_id).to(attentions) # [N, ]
        indices = [node.id for node in nodes]
        similarity[indices] = attentions # fill the blanks

        dist = similarity @ self.adj_matrix.to(similarity)

        self.log_str(similarity, tag="Similarity")
        self.log_str(dist, tag="Dist")

        # find maximum
        dist[list(memory_ids)] = 0.0 # mask the chosen ones
        max_idx = torch.argmax(dist).item()

        self.log_str(f"Search to node: {self.id2node[max_idx].desc}(id = {self.id2node[max_idx].id}), score = {dist[max_idx]}")

        return self.id2node[max_idx]

    def query(self, prompt: str, ignore_charts: bool=True, switch_set: bool=False) -> str:
        tries = 0
        # TODO
        memory: List[GarlicNode] = self.layers[-1]
        if ignore_charts:
            memory = [node for node in memory if len(node.children)]

        memory_ids = set([node.id for node in memory])
        cur_image = None
        finish_result, attention_score = self.finish_query(prompt, memory, multi_models=True)
        self.log_str(f"<Initial info>\n" + "".join(self._fetch_desc(memory)) + "</Initial info>\n")
        while 'yes' not in finish_result.lower() and tries <= self.max_tries:
            self.log_str(finish_result, tag="Finish result")

            next_node = self.graph_search(memory, memory_ids, attention_score)
        
            memory.append(next_node)
            memory_ids.add(next_node.id)

            finish_result, attention_score = self.finish_query(prompt, memory, multi_models=True)

            tries += 1

            cur_image = None

        memories = "\n".join(reversed(self._fetch_desc(memory)))

        return self.lmodel.predict(f"Answer the question: {prompt}, and give your evidence, with the following information:\n{memories}")