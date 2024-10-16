"""
NOTE: llava-ov-0.5b can run without flash-attn; But it seems that llava-ov-7b cannot.
And attn scores is not compatible with flash-attn, so attn can only run with 0.5b model
"""
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import copy
import torch
from typing import Union, List
from tqdm import tqdm
import sys
import warnings

from models.vlm import VLMMeta, ImageLike, preprocess

class LlavaOV(VLMMeta):
    def __init__(self, model_path: str,
                       model_name: str="llava_qwen",
                       device_map: str="auto",
                       max_new_tokens: int=8192):
        warnings.filterwarnings("ignore")
        self.device = device_map
        self.max_new_tokens = max_new_tokens
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(model_path, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
        self.model.eval()
    
    def predict(self, prompt: str, image: Union[ImageLike, None]=None) -> str:
        if image is not None:
            if type(image) == list:
                image_tensor = process_images(image, self.image_processor, self.model.config)
            else:
                image = Image.open(image) if type(image) == str else image
                image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        conv_template = "qwen_1_5"
        if image is not None:
            question = (DEFAULT_IMAGE_TOKEN + "\n") + prompt
        else:
            question = prompt
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        if image is not None:
            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            image_sizes = [image.size] if type(image) != list else [img.size for img in image]
        else:
            input_ids = self.tokenizer([prompt_question], return_tensors="pt")['input_ids'].to(self.device)
            image_tensor = image_sizes = None

        with torch.no_grad():
            cont = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=self.max_new_tokens
            )

        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        
        return text_outputs[0]