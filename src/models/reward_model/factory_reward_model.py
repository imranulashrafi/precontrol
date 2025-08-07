import torch
import torch.nn as nn
from typing import Optional, List
from .modeling_ultrarm import LlamaRewardModel
from transformers import LlamaTokenizer


class RewardModel:
    def __init__(self, reward_model_name):
        self.reward_model_name = reward_model_name
        if reward_model_name == "openbmb/UltraRM-13b":
            self.model = LlamaRewardModel.from_pretrained(
                reward_model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to("cuda:1")
            self.tokenizer = LlamaTokenizer.from_pretrained(
                reward_model_name, use_fast=True
            )
        self.model.eval()

    def score(self, prompt, response):
        if self.reward_model_name == "openbmb/UltraRM-13b":
            prompt = prompt.replace("User:", "Human:")
            encoded = self.tokenizer(
                prompt + response,
                return_tensors="pt",
            ).to("cuda:1")
        with torch.no_grad():
            outputs = self.model(**encoded)
        return outputs
