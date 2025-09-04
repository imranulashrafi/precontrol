import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from src.models.intervened_model.factory_intervened_model import IntervenedLM
from math import ceil


class InferenceIntervention:
    def __init__(
        self,
        model_path,
        tokenizer,
        use_intervention=True,
        value_model=None,
        epochs=50,
        lr=0.5,
    ):
        self.model_path = model_path
        self.use_intervention = use_intervention
        self.value_model = value_model
        self.epochs = epochs
        self.lr = lr

        self.model = None
        self.tokenizer = tokenizer

        self._load_model_and_tokenizer()

    def generate_and_decode(self, batch, prompt_batch):
        generated_responses = []

        device = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
        )
        
        decoded = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        for prompt, response in zip(prompt_batch, decoded):
            result = response.removeprefix(prompt)
            generated_responses.append(
                {"prompt": prompt, "result": result, "response": response}
            )

        return generated_responses

    def _load_model_and_tokenizer(self):
        if self.use_intervention:
            # print("coming here?")
            self.model = IntervenedLM.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model.set_value_model(self.value_model)
            self.model.set_lr_and_epochs(self.lr, self.epochs)
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.do_sample = False
        self.model.generation_config.num_beams = 1
        self.model.config.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
