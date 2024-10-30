import json
import torch
import pickle
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Any, Set
from sentence_transformers import SentenceTransformer

from models.base import VLMBase, ImageLike, LLMBase
from structure.tree import TreeBase

class GarlicNode:
    node_id = 0
    ir_model = SentenceTransformer("/share_io02_hdd/wangyaoning/sbert")
    children: List[Any]

    def __init__(self, image: Union[ImageLike, None]=None,
                       desc: Union[str, None]=None,
                       page_num: int=-1) -> None:
        self.image = image
        self.desc = desc
        self.children = []
        self.page_num = page_num
        self.embedding = GarlicNode.ir_model.encode(self.desc, convert_to_tensor=True)

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

class GarlicTree(TreeBase):
    model: VLMBase
    def __init__(self,
                 vmodel: VLMBase,
                 log: str=None,
                 max_tries: int=10,
                 late_stops: int=5,
                 look_image: bool=True):
        self.model = vmodel
        self.max_tries = max_tries
        self.late_stops = late_stops
        self.look_image = look_image
        self.prompt_template = "Summary the following information. Split your summary into different summary points according to the semantic information in these information points. It is not necessary to generate each summary point for each information point. Gather and organize information into summary points. In each summary point, try to avoid using pronouns like he/she/they and instead use full names. Generate in the format of: * <summary point1>\n* <summary point2>\n* <summary point3>."
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
    
    def as_json(self, file_path: str, init: Union[List[int], None]=None, traj: Union[List[int], None]=None) -> None:
        obj = {
            "nodes": [],
            "edges": []
        }
        for i, layer in enumerate(self.layers):
            for node in layer:
                obj["nodes"].append({
                    "id": node.id,
                    "desc": node.desc,
                    "layer": i,
                    "from": node.page_num
                })
                for (child, weight) in node.children:
                    obj["edges"].append([node.id, child.id, weight])
        
        if init:
            obj["init"] = init
        if traj:
            obj["traj"] = traj
        
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
            
            # embeddings
            self.embeddings = []
            for i in range(GarlicNode.node_id):
                self.embeddings.append(self.id2node[i].embedding)

            self.embeddings = torch.stack(self.embeddings)
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
            self.model.pre_prompts + self.get_prompt_template(layer),
            *texts,
            self.model.post_prompts
        ]
        self.log_str("Raw inputs: \n" + str(model_inputs) + "\n")
        new_text, attentions = self.model.attentioned_predict("".join(model_inputs))
        self.log_str(f"Raw prediction: {new_text}")
        IPs = self._parse_response(new_text)
        scores = self.model.str_level_score(model_inputs, IPs, attentions) # [len(IPs), len(nodes) + 2]
        scores = scores[:, 1: -1] # remove chat template
        self.log_str(scores, tag="Edge Weight")
        
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
        
        def filter_position(positions: List[int]) -> List[int]:
            filtered = list()
            for pos in positions:
                if len(filtered) == 0:
                    filtered.append(pos)
                    continue
                if pos - filtered[-1] <= 5:
                    continue
                filtered.append(pos)
            return filtered
        
        self.log_str(positions, tag="Before filter")

        positions = filter_position(positions)

        self.log_str(positions, tag="After filter")

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
    
    def select_page_num(self, nodes: List[GarlicNode], idx: int, token_sum: int=3000) -> int:
        approx_token_lens = [len(node.desc) for node in nodes]
        approx_token_lens = [self.model.token_len(node.desc) for node in nodes]
        token_len, page_num = approx_token_lens[idx], 1
        while token_len < token_sum and idx + page_num < len(nodes):
            token_len += approx_token_lens[idx + page_num]
            page_num += 1
        page_num = max(2, page_num)
        self.log_str(f"Select {page_num} pages, approx token = {token_len}\n")
        
        return page_num

    def build_text(self, passage: str, path: str="") -> None:
        """
        The basic function of Garlic Tree.
        """
        GarlicNode.node_id = 0
        self.layers = list()
        leaves = list()

        tokens = torch.split(self.model.tokenize(passage), 300, dim=1)

        for chunk in tokens:
            leaves.append(GarlicNode(desc=self.model.detokenize(chunk)))
            
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

        self.id2node = dict()
        for layer in self.layers:
            for node in layer:
                self.id2node[node.id] = node
        
        self.embeddings = []
        for i in range(GarlicNode.node_id):
            self.embeddings.append(self.id2node[i].embedding)

        self.embeddings = torch.stack(self.embeddings)
        
        # calculate adj matrix
        self.adj_matrix = torch.zeros((GarlicNode.node_id, GarlicNode.node_id), dtype=torch.float32)
        for layer in self.layers:
            for node in layer:
                node: GarlicNode
                ids, weights = self._fetch_weights(node)
                if len(ids):
                    self.adj_matrix[node.id][ids] = weights
        
        self.log_str(str(self.adj_matrix))

        print(self.layers)

        self._save_tree(path + "_tree.pkl")
    
    def build(self, images: ImageLike, leaves: str=None, path: str="") -> None:
        """
        The basic function of Garlic Tree.
        """
        GarlicNode.node_id = 0
        self.layers = list()
        self.image_descs = list()
        leaves = list() if leaves is None else self._load_leaves(leaves)

        if len(leaves) == 0:
            for i, image in enumerate(images):
                w, h = image.size
                self.log_str(f"Predicting page {i}...\n")
                response = self.model.predict(self.get_prompt_template(0), image, return_attn=False)
                IPs = self._parse_response(response)
                leaves.extend([GarlicNode(image=image, desc=ip, page_num=i) for ip in IPs])
                # leaves.append(GarlicNode(image=image, desc=self.model.predict(self.get_prompt_template(0), image))) # Original
                # self.image_descs.append(GarlicNode(image=image, desc=self.model.predict(self.prompt_template_img, image)))
                self.log_str(f"Summarize of page {i}: \n***************************************************************************************************\n{leaves[-1].desc}\n")
                # self.log_str(f"Summarize of figure: \n{self.image_descs[-1].desc}\n***************************************************************************************************\n")

            self._save_leaves(leaves, path + "_leaves.pkl")
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
        
        self.embeddings = []
        for i in range(GarlicNode.node_id):
            self.embeddings.append(self.id2node[i].embedding)

        self.embeddings = torch.stack(self.embeddings)
        
        # calculate adj matrix
        self.adj_matrix = torch.zeros((GarlicNode.node_id, GarlicNode.node_id), dtype=torch.float32)
        for layer in self.layers:
            for node in layer:
                node: GarlicNode
                ids, weights = self._fetch_weights(node)
                if len(ids):
                    self.adj_matrix[node.id][ids] = weights
        
        self.log_str(str(self.adj_matrix))

        self._save_tree(path + "_tree.pkl")
        
    def finish_query(self, prompt: str, memory: List[GarlicNode], images: ImageLike=None) -> Tuple[str, torch.Tensor]:
        """
        Given prompt and memory, ask LLM(VLM) if the question can be answered.
        """
        template = [self.model.pre_prompts,
                    f"Question: {prompt}\n\n",
                    "Can this question be answered by the following information? Response \"Yes\" or \"No\" in one word, then provide your reason.\n\n",
                    f"Information:\n",
                    *[node.desc.replace('<|im_end|>', '') for node in memory],
                    self.model.post_prompts
                    ]
        if not images or not self.look_image:
            self.log_str(f"Raw finish input: {template}\n")
            output, attentions = self.model.attentioned_predict("".join(template), inner_attention=True, max_new_tokens=100)
            scores = self.model.str_level_score(template, template, attentions, io_attention=False) # [5 + len(memory), 5 + len(memory)]
            self.log_str(str(scores), tag="Score before adjusting")
            scores = scores[4: -1, 1] # [len(memory)]
            # compensation on position
            scores = scores * torch.arange(1, len(memory) + 1).to(scores)
            return output, scores
        
        # image
        output, attentions = self.model.attentioned_predict("".join(template[1:-1]), images, inner_attention=True, output_only=False)
        pre_prompts = output[:len(self.model.pre_prompts)]
        prompt_idx = output.find("".join(template[1:-1]))
        image_prompts = output[len(self.model.pre_prompts): prompt_idx]
        final_output = output[prompt_idx + len("".join(template[1:-1]))]
        input_seq = [
            pre_prompts,
            image_prompts,
            *template[1:-1],
            self.model.post_prompts
        ]
        scores = self.model.str_level_score(
            input_seq,
            input_seq,
            attentions=attentions,
            io_attention=False
        )
        scores = scores[5:-1, 2]
        scores = scores * torch.arange(1, len(memory) + 1).to(scores)
        return output, scores

    def graph_search(self, nodes: List[GarlicNode], memory_ids: Set[int], attentions: torch.Tensor, query_embedding: torch.Tensor) -> GarlicNode:
        """
        Search the next node to be queried.
        NOTE: It seems that similarity norm is not so reasonable. Maybe larger similarity should be searched first?
        NOTE: Maybe a concentration mechanism should be adapted. Now this search method tends to distract by irrelevant IPs.
        """
        # construct query-node similarity vector
        similarity = torch.zeros(GarlicNode.node_id).to(attentions) # [N, ]
        indices = [node.id for node in nodes]
        similarity[indices] = attentions # fill the blanks

        dist = similarity @ self.adj_matrix.to(similarity)

        dist /= dist.sum() # normalize
        self.log_str(dist, tag="Dist")
        # sbert similarity
        sbert_sim = GarlicNode.ir_model.similarity(torch.from_numpy(query_embedding).squeeze(0).to(dist), self.embeddings.to(dist))

        dist += sbert_sim.flatten()

        self.log_str(similarity, tag="Similarity")
        self.log_str(dist, tag="Final Dist")

        # find maximum
        dist[list(memory_ids)] = 0.0 # mask the chosen ones
        max_idx = torch.argmax(dist).item()

        self.log_str(f"Search to node: {self.id2node[max_idx].desc}(id = {self.id2node[max_idx].id}), score = {dist[max_idx]}")
        return self.id2node[max_idx]

    def late_stop(self) -> bool:
        self.late_stop_counter -= 1
        return self.late_stop_counter > -1

    def query(self, prompt: str, ignore_charts: bool=True, export_traj: Union[None, str]=None) -> str:
        tries = 0
        self.late_stop_counter = self.late_stops
        query_embedding = GarlicNode.ir_model.encode(prompt)
        # TODO
        memory: List[GarlicNode] = self.layers[-1]

        # if ignore_charts:
        #     memory = [node for node in memory if len(node.children)]
        
        if export_traj:
            init = [node.id for node in memory]
            traj = list()

        memory_ids = set([node.id for node in memory])
        cur_images = list()
        finish_result, attention_score = self.finish_query(prompt, memory, cur_images)
        self.log_str(f"<Initial info>\n" + "".join(self._fetch_desc(memory)) + "</Initial info>\n")
        while ('yes' not in finish_result.lower() or self.late_stop()) and tries <= self.max_tries and len(memory) < GarlicNode.node_id:
            self.log_str(finish_result, tag="Finish result")

            next_node = self.graph_search(memory, memory_ids, attention_score, query_embedding)

            if isinstance(next_node.image, Image.Image):
                cur_images.append(next_node.image)
        
            memory.append(next_node)
            memory_ids.add(next_node.id)
            if export_traj:
                traj.append(next_node.id)

            finish_result, attention_score = self.finish_query(prompt, memory, cur_images)

            tries += 1

        memories = "\n".join(self._fetch_desc(memory)) # REVERSED?

        self.log_str(f"Exit after {tries} searches.\n")

        if export_traj:
            self.as_json(export_traj, init, traj)

        return self.model.predict(self.model.pre_prompts + f"Given the above information and question, answer the question as concisely as you can.\nQuestion: \n{prompt}\n\nInformation:\n{memories}" + self.model.post_prompts,
                                  image=cur_images if len(cur_images) else None,
                                  return_attn=False)