import json
import torch
import pickle
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Any

from models.vlm import VLMMeta, ImageLike, preprocess, LLM
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
    def __init__(self, vmodel: VLMMeta, lmodel: Union[VLMMeta, LLM], log: str=None):
        self.model = vmodel
        self.lmodel = lmodel
        self.prompt_template = "Please read and summarize the following paragraphs carefully, Split your summary into different summary points according to the semantic information in these information points. It is not necessary to generate each summary point for each information point. Gather and organize information into summary points. In each summary point, try to avoid using pronouns like he/she/they and instead use full names. Generate in the format of: * <summary point1>\n* <summary point2>\n* <summary point3>"
        self.interm_template = "Please group the summary points, for each group, give it a subtitle and briefly describe it. Do not repeat the information point in the given paragraphs. Do not list any specific figure or number(For example, percentage / number...). Generate in the format of: **subtitle 1**: <description of subtitle1>\n**subtitle 2**: <description of subtitle2>\n..."
        self.prompt_template_img = "Are there any figures(pie chart, table, bar chart, etc.) in this document? If so, briefly summarize meaning of each figure in this page. Else, reply a single string `No`. Do not list any specific number or value."
        if log:
            self.log = log
            with open(self.log, "w") as fp:
                fp.close()
    
    def log_str(self, text: str, newline: int=200) -> None:
        text = str(text)
        def insert_newline(s, n):
            return '\n'.join([s[i:i+n] for i in range(0, len(s), n)])
        with open(self.log, "a") as fp:
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
        with open(file_path, "rb") as fp:
            self.layers = pickle.load(fp)
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
    
    def _fetch_weights(self, nodes: List[GarlicNode], return_tensors: str="pt") -> Union[torch.Tensor, np.ndarray]:
        assert return_tensors in ["pt", "np"]
        # TODO
    
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
        texts = self._fetch_desc(nodes)
        model_inputs = [
            self.lmodel.pre_prompts + self.get_prompt_template(layer),
            *texts,
            self.lmodel.post_prompts
        ]
        self.log_str("Raw inputs: \n" + "".join(model_inputs) + "\n")
        new_text, attentions = self.lmodel.attentioned_predict("".join(model_inputs))
        self.log_str(f"Raw prediction: {new_text}")
        IPs = self._parse_response(new_text)
        scores = self.lmodel.str_level_score(model_inputs, IPs, attentions) # [len(IPs), len(nodes) + 2]
        scores = scores[:, 1: -1] # remove chat template
        
        assert len(IPs) == scores.shape[0]
        assert len(nodes) == scores.shape[1]

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
        self.leaf_num = len(leaves)

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
        
        self._save_tree("tree.pkl")
        
    def finish_query(self, prompt: str, memory: List[GarlicNode], images: ImageLike=None) -> bool:
        memories = "\n".join(reversed(self._fetch_desc(memory)))
        self.log_str(f"CURRENT LENGTH = {len(memories)}\n")
        template = f"Can this question be answered by the following information? Response Yes or No in one word. Do not provide any explanation or directly answer the question. Question: {prompt}\nInformation:\n{memories}"
        return self.model.predict(template, images)

    def query(self, prompt: str) -> str:
        # TODO
        memory = self.layers[-1]
        memory_ids = set([node.id for node in memory])
        cur_image = None
        finish_result = self.finish_query(prompt, memory, images=cur_image)
        self.log_str(f"<Initial info>\n" + "\n".join(self._fetch_desc(memory)) + "</Initial info>\n")
        while 'yes' not in finish_result.lower():
            self.log_str(f"Finish result: {finish_result}\n")

            # expand
            next_nodes = list()
            for node in memory:
                for next_node in node.children:
                    if next_node.id not in memory_ids:
                        next_nodes.append(next_node)
            
            result = self.model.predict(self.get_query_template(prompt, next_nodes)).strip()
            expand_id = int(result) - 1
            memory.append(next_nodes[expand_id])
            self.log_str(f"<Expand>\n{next_nodes[expand_id].desc}</Expand>")
            if next_nodes[expand_id].image is not None:
                cur_image = next_nodes[expand_id].image

            finish_result = self.finish_query(prompt, memory, images=cur_image)

            cur_image = None
        
        memories = "\n".join(reversed(self._fetch_desc(memory)))

        return self.model.predict(f"Answer the question: {prompt}, and give your evidence, with the following information:\n{memories}", cur_image)
    
    def retrieve(self, prompt: str) -> List[int]:
        # TODO
        pass