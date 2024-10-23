"""
Warning: paligemma seems not able to return attn_scores
"""
import torch
from typing import Union, List, Tuple
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaConfig

from models.meta import VLMMeta, ImageLike, preprocess

class Paligemma(VLMMeta):
    def __init__(self, model_path: str, device: str="cuda", max_new_tokens: int=8192):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, device_map=device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
    
    def predict(self, prompt: Union[str, None]=None, image: ImageLike=None) -> str:
        if image is not None:
            image = preprocess(image)
        
        model_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            
        return decoded
    
    def tokenized_length(self, prompt: str) -> int:
        return self.processor.tokenizer([prompt], return_tensors="pt")["input_ids"].shape[-1]
    
    def text_only_predict(self, prompt: str, return_attn=False) -> Union[str, Tuple[str, torch.Tensor]]:
        """
        Do text-only prediction.
        Output:
        - When return_attn set to `True`, return (prediction, attentions), where
        `attentions` is a tuple of tensor [bs, num_heads, seq_len, seq_len].
        - else, return a single prediction.
        """
        model_inputs = self.processor.tokenizer([self.processor.tokenizer.bos_token + prompt + "\n"], return_tensors="pt").to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            if return_attn:
                generation = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, do_sample=False, output_attentions=True, return_dict_in_generate=True)
                seq = generation.sequences
            else:
                generation = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                seq = generation

            seq = seq[0][input_len:]
            decoded = self.processor.decode(seq, skip_special_tokens=True)
        
        if return_attn:
            return decoded, generation.attentions[0] # Tuple[Tuple[Tensor, ...]]

        return decoded
    
    def blocked_text_attention(self, texts: List[str], join_str: str="") -> torch.Tensor:
        """
        Return normalized, block-averaged text attention from `texts`.
        Outputs:
        Tuple[str, Tensor], where:
        str: generation str.
        Tensor: attention scores of shape [len(texts),] indicating the weights between output and each text in `texts`.
        """
        lengths = [self.tokenized_length(text) for text in texts]
        # lengths: lengths after tokenization
        output, attentions = self.text_only_predict(prompt=join_str.join(texts), return_attn=True)
        x = torch.stack(attentions) # [18, bs, 8, 1139, 1139]
        print(len(output))

    def get_attention(self, image: ImageLike) -> torch.Tensor:
        """
        Args:
        - image: ImageLike object, include `bs` images.
        Output:
        Tuple of torch.Tensor, each tensor has shape [bs, num_heads, seq_len, seq_len].
        By default paligemma(precisely SigLIP), `num_heads` = 16, `seq_len` = 1024; #layer = 27. 
        """
        image = preprocess(image)
        image = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            image_outputs = self.model.vision_tower(image["pixel_values"], output_attentions=True)
        return image_outputs.attentions # attention: 27 layers, each [bs, 16, 1024, 1024] -> [bs, num_heads, seq_len, seq_len]
    
    def get_projected_attention(self, prompt: Union[str, None]=None, image: Union[ImageLike, None]=None) -> torch.Tensor:
        image = preprocess(image)
        model_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.forward(**model_inputs, output_attentions=True)
        return outputs.attentions # [bs, num_heads, seq_len, seq_len]