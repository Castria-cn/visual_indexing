import torch
from typing import List, Tuple
from models.meta import LLMMeta
from transformers import Qwen2ForCausalLM, AutoTokenizer

class Qwen(LLMMeta):
    def __init__(self, model_path: str, device_map: str="cuda", max_new_tokens: int=2048, **kwargs):
        self.device = device_map
        self.max_new_tokens = max_new_tokens
        self.model = Qwen2ForCausalLM.from_pretrained(model_path, device_map=device_map, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self.pre_prompts = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        self.post_prompts = "<|im_end|><|im_start|>assistant\n"
    
    def predict(self, prompt: str, return_attn: bool = False) -> str | Tuple[str, Tuple, int]:
        """
        When `return_attn` set to `True`, returns a tuple[str, Tuple, int];
        where:
        - str: The **WHOLE** output sequence(include input seq);
        - Tuple(named `attn` here): A tuple, each element also a tuple. attn[i] indicates the i-th token's attention score.
          For i == 0, it is the input tokens' attentions, thus form a tuple of length 16(attn layer num), each element of shape [1, num_heads=32, l, l];
          For i >= 1, it is the output token attentions, thus form a tuple of length 16(attn layer num), each element of shape [1, num_heads=32, 1, l + i].
          note that, for each tensor in the tuple, tensor.sum(-1) == 1.
        - int: The input sequence length `l`(token level).
        NOTE: For QWEN, we retain the special tokens (<|im_start|>, <|im_end|>)
        """
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]

        if not return_attn:
            outputs = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][input_len:]
        # return attn
        outputs = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, output_attentions=True, return_dict_in_generate=True, do_sample=False)

        return (self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[0], outputs.attentions, input_len)

    def attentioned_predict(self, prompt: str, output_only: bool=True, inner_attention: bool=False) -> Tuple[str, torch.Tensor]:
        """
        When `inner_attention` set to `True`, returns (output, attention scores in input sequence). 
        - str: The predicted result.
        - torch.Tensor: The token level attention scores with the input, which has the shape [output_len, input_len]
        NOTE: input_len DOES NOT contains bos token.
        """
        outputs, attentions, input_len = self.predict(prompt, return_attn=True)

        if inner_attention:
            attentions = attentions[0] # Tuple[tensor[1, 32, l, l]]
            attentions = torch.stack(attentions) # [16, 1, 32, l, l]
            attentions = attentions.mean(0).squeeze(0).mean(0) # [l, l] bos not removed.
            return outputs[len(prompt) if output_only else None:], attentions

        output_token_attentions = [torch.stack(attn).mean(0).squeeze(0).mean(0).squeeze(0) for attn in attentions[1:]] # each element [l + pos, ]. attentions[1:]: remove the input-inner-attentions.
        
        """
        NOTE:
        Maybe some normalize should be done here, but the GARLIC paper did not mention it.
        For example, input_len = 10, the third output token's attention has the shape [13, ]
        To get the attention score with attention[:10], we have to cancel the effect of the last 3 tokens. But softmax operator is not invertable.

        Or, we can simply retain the edges in the outputs?

        Now we select the first method: normalize the first k components by dividing their sum.
        """
        output_token_attentions = torch.stack([attn[1:input_len] / attn[1:input_len].sum() for attn in output_token_attentions]) # [output_tokens, input_len]. attn[1: input_len]: remove the input bos token.

        return outputs[len(prompt) if output_only else None:], output_token_attentions

    def align_token_str(self, splited: List[str], tokens: torch.Tensor) -> List[Tuple[int, int]]:
        """
        For each chunk in `splited`, find the [start, end) of it in `tokens`.
        NOTE: tokenize("".join(splited)) == tokens
        Outputs:
        List[Tuple[int, int]]: each element a [start, end) tuple.
        
        """
        # Step 1. Align splited with original string
        offsets: List[Tuple[int, int]] = [(0, len(splited[0]))]

        for chunk in splited[1:]:
            start = offsets[-1][1]
            end = start + len(chunk)
            offsets.append((start, end))

        # Step 2. Align original string with tokens
        full = "".join(splited)
        token_offsets = self.tokenizer(full, return_offsets_mapping=True)["offset_mapping"][1:] # remove bos
        token_offsets: List[Tuple[int, int]] = [(token_offsets[i][0], token_offsets[i + 1][0] if i < len(token_offsets) - 1 else len(token_offsets)) for i in range(len(token_offsets))]

        # Step 3. Align splited with tokens
        """
        chunk_start   chunk_end
        [             |              |          |    ]
        start   end
        [       |                                    ] full
        |      /
        |     /
        [    |                   ] tokens
        """
        chunk_ptr = 0
        chunk_align = [(len(tokens), 0)] * len(splited)
        for i, (start, end) in enumerate(token_offsets): # for each token, check if it is in chunk
            if i and (start >= chunk_end): # move to next chunk
                chunk_ptr += 1
            if chunk_ptr >= len(splited):
                break
            chunk_start, chunk_end = offsets[chunk_ptr]
            if end > chunk_start and end <= chunk_end: # token[i] in chunk[chunk_ptr]
                start_ = min(chunk_align[chunk_ptr][0], i)
                end_ = max(chunk_align[chunk_ptr][1], i + 1)
                chunk_align[chunk_ptr] = (start_, end_)
        
        # print("Align result:")
        # for i, (start, end) in enumerate(chunk_align):
        #     seq = tokens[start: end]
        #     print(f"[{splited[i]}] aligned to [{self.tokenizer.batch_decode(seq)}]")
        
        return chunk_align
    
    def str_level_score(self, splited_input: List[str], splited_output: List[str], attentions: torch.Tensor, io_attention: bool=True) -> torch.Tensor:
        """
        - splited_input: The splited format of the input. e.g. ["What is the derivative of function", "f(x) = e^x?"]
        - splited_output: The splited format of the output. e.g. ["The derivative of function f(x) = e^x is ] NOTE: Not include input sequence.
        - attentions: tensor of shape [output_len, input_len].
        - io_attention: set to `True` when calculating input-output attention. When calculating input-input attention, set to `False`.
        NOTE: splited_input, splited_output must be same as input, output.
        """
        input = self.tokenizer("".join(splited_input), return_tensors="pt")["input_ids"][0].to(self.device)
        output = self.tokenizer("".join(splited_output), return_tensors="pt")["input_ids"][0].to(self.device)

        assert input.shape[-1] - io_attention == attentions.shape[-1]
        assert output.shape[-1] - io_attention == attentions.shape[0]

        input_mapping, output_mapping = self.align_token_str(splited_input, input), self.align_token_str(splited_output, output) # input_mapping: [start, end) of splited_input[i] in the whole input

        # calculate blocked attention score
        scores = list()
        for i, (output_start, output_end) in enumerate(output_mapping):
            input_scores = list()
            for j, (input_start, input_end) in enumerate(input_mapping):
                block = attentions[output_start: output_end, input_start: input_end]
                input_scores.append(block.mean(0).sum().item())
            scores.append(input_scores)
        
        scores = torch.tensor(scores).to(self.device)

        return scores