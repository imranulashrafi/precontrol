import os
import yaml

with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["HF_HOME"] = config["hf_home"]

import torch
import h5py
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src.utils.activations_generator import ActivationsGenerator
import numpy as np
import hdf5plugin
from numcodecs import Blosc

CONFIG_PATH = "experiments/config.yaml"
BATCH_SIZE = 8
MAX_SEQ_LEN = 256
MAX_INPUT_TOKENS = 300
OUTPUT_DIR = "/home"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_inputs_batch(batch, tokenizer):
    def build_input(histories, answers):
        return [f"User: {h}\nAssistant: {a}" for h, a in zip(histories, answers)]

    labels = batch["labels"]
    A = batch["human_ref_A"]
    B = batch["human_ref_B"]
    H = batch["history"]

    preferred = build_input(H, [a if l == 1 else b for l, a, b in zip(labels, A, B)])
    rejected = build_input(H, [b if l == 1 else a for l, a, b in zip(labels, A, B)])
    generated = [f"User: {h}\nAssistant:" for h in H]

    encoded_generated = tokenizer(generated, padding=False, truncation=True)
    encoded_preferred = tokenizer(preferred, padding=False, truncation=True)
    encoded_rejected = tokenizer(rejected, padding=False, truncation=True)

    return {
        "prompt_generated_input_ids": encoded_generated["input_ids"],
        "prompt_generated_attention_mask": encoded_generated["attention_mask"],
        "prompt_preferred_input_ids": encoded_preferred["input_ids"],
        "prompt_preferred_attention_mask": encoded_preferred["attention_mask"],
        "prompt_rejected_input_ids": encoded_rejected["input_ids"],
        "prompt_rejected_attention_mask": encoded_rejected["attention_mask"],
        "text_generated": generated,
        "text_preferred": preferred,
        "text_rejected": rejected,
        "len_generated": [len(ids) for ids in encoded_generated["input_ids"]],
        "len_preferred": [len(ids) for ids in encoded_preferred["input_ids"]],
        "len_rejected": [len(ids) for ids in encoded_rejected["input_ids"]],
    }


def streaming_collate_fn(batch):
    def extract(field_prefix):
        return tokenizer.pad(
            {
                "input_ids": [ex[f"{field_prefix}_input_ids"] for ex in batch],
                "attention_mask": [
                    ex[f"{field_prefix}_attention_mask"] for ex in batch
                ],
            },
            return_tensors="pt",
        )

    return {
        "tokenized_prompts": extract("prompt_generated"),
        "tokenized_preferred": extract("prompt_preferred"),
        "tokenized_rejected": extract("prompt_rejected"),
        "prompts_generated": [ex["text_generated"] for ex in batch],
        "prompts_preferred": [ex["text_preferred"] for ex in batch],
        "prompts_rejected": [ex["text_rejected"] for ex in batch],
    }


def pad_tensor(x, max_len):
    pad_len = max_len - x.shape[1]
    if pad_len > 0:
        pad = torch.zeros(
            x.shape[0], pad_len, x.shape[2], dtype=x.dtype, device=x.device
        )
        x = torch.cat([x, pad], dim=1)
    return x[:, :max_len]


def pad_mask(x, max_len):
    pad_len = max_len - x.shape[1]
    if pad_len > 0:
        pad = torch.zeros(x.shape[0], pad_len, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=1)
    return x[:, :max_len]


class FastH5Writer:
    def __init__(self, path, total_samples, seq_dim, hidden_dim):
        self.index = 0
        self.total_samples = total_samples

        self.f = h5py.File(path, "w", libver="latest")

        self.datasets = {
            "generated/activations": self.f.create_dataset(
                "generated/activations",
                (total_samples, seq_dim, hidden_dim),
                dtype="float16",
            ),
            "generated/masks": self.f.create_dataset(
                "generated/masks", (total_samples, seq_dim), dtype="uint8"
            ),
            "generated/responses": self.f.create_dataset(
                "generated/responses",
                (total_samples,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            ),
            "preferred/activations": self.f.create_dataset(
                "preferred/activations",
                (total_samples, seq_dim, hidden_dim),
                dtype="float16",
            ),
            "preferred/masks": self.f.create_dataset(
                "preferred/masks", (total_samples, seq_dim), dtype="uint8"
            ),
            "rejected/activations": self.f.create_dataset(
                "rejected/activations",
                (total_samples, seq_dim, hidden_dim),
                dtype="float16",
            ),
            "rejected/masks": self.f.create_dataset(
                "rejected/masks", (total_samples, seq_dim), dtype="uint8"
            ),
        }

    def write_batch(self, batch_dict):
        s = self.index
        e = s + batch_dict["generated_activations"].shape[0]

        self.datasets["generated/activations"][s:e] = self._np(
            batch_dict["generated_activations"]
        )
        self.datasets["generated/masks"][s:e] = self._np(
            batch_dict["generated_masks"], dtype="uint8"
        )
        self.datasets["generated/responses"][s:e] = self._encode_strs(
            batch_dict["generated_responses"]
        )

        self.datasets["preferred/activations"][s:e] = self._np(
            batch_dict["preferred_activations"]
        )
        self.datasets["preferred/masks"][s:e] = self._np(
            batch_dict["preferred_masks"], dtype="uint8"
        )
        self.datasets["rejected/activations"][s:e] = self._np(
            batch_dict["rejected_activations"]
        )
        self.datasets["rejected/masks"][s:e] = self._np(
            batch_dict["rejected_masks"], dtype="uint8"
        )

        self.index = e

    def _np(self, data, dtype=None):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if dtype:
            data = data.astype(dtype)
        return data

    def _encode_strs(self, texts):
        return np.array(texts, dtype=h5py.string_dtype(encoding="utf-8"))

    def close(self):
        self.f.flush()
        self.f.close()


class FastH5WriterCompress:
    def __init__(self, path, total_samples, seq_dim, hidden_dim):
        self.index = 0
        self.total_samples = total_samples

        self.f = h5py.File(path, "w", libver="latest")

        # Blosc compression options
        blosc_opts = hdf5plugin.Blosc(
            cname="lz4", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
        )

        def create(name, shape, dtype, compress=False):
            if len(shape) == 3:
                chunk_shape = (min(128, shape[0]), shape[1], shape[2])
            elif len(shape) == 2:
                chunk_shape = (min(128, shape[0]), shape[1])
            else:
                chunk_shape = (min(128, shape[0]),)
            return self.f.create_dataset(
                name,
                shape=shape,
                dtype=dtype,
                chunks=chunk_shape,
                **(blosc_opts if compress else {}),
            )

        self.datasets = {
            "generated/activations": create(
                "generated/activations",
                (total_samples, seq_dim, hidden_dim),
                "float16",
                compress=True,
            ),
            "generated/masks": create(
                "generated/masks", (total_samples, seq_dim), "uint8"
            ),
            "generated/responses": self.f.create_dataset(
                "generated/responses",
                (total_samples,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            ),
            "preferred/activations": create(
                "preferred/activations",
                (total_samples, seq_dim, hidden_dim),
                "float16",
                compress=True,
            ),
            "preferred/masks": create(
                "preferred/masks", (total_samples, seq_dim), "uint8"
            ),
            "rejected/activations": create(
                "rejected/activations",
                (total_samples, seq_dim, hidden_dim),
                "float16",
                compress=True,
            ),
            "rejected/masks": create(
                "rejected/masks", (total_samples, seq_dim), "uint8"
            ),
        }

    def write_batch(self, batch_dict):
        s = self.index
        e = s + batch_dict["generated_activations"].shape[0]

        self.datasets["generated/activations"][s:e] = self._np(
            batch_dict["generated_activations"]
        )
        self.datasets["generated/masks"][s:e] = self._np(
            batch_dict["generated_masks"], dtype="uint8"
        )
        self.datasets["generated/responses"][s:e] = self._encode_strs(
            batch_dict["generated_responses"]
        )

        self.datasets["preferred/activations"][s:e] = self._np(
            batch_dict["preferred_activations"]
        )
        self.datasets["preferred/masks"][s:e] = self._np(
            batch_dict["preferred_masks"], dtype="uint8"
        )
        self.datasets["rejected/activations"][s:e] = self._np(
            batch_dict["rejected_activations"]
        )
        self.datasets["rejected/masks"][s:e] = self._np(
            batch_dict["rejected_masks"], dtype="uint8"
        )

        self.index = e

    def _np(self, data, dtype=None):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if dtype:
            data = data.astype(dtype)
        return data

    def _encode_strs(self, texts):
        return np.array(texts, dtype=h5py.string_dtype(encoding="utf-8"))

    def close(self):
        self.f.flush()
        self.f.close()


def process_split(split_name, raw_dataset, activation_generator, tokenizer, config):
    print(f"Processing split: {split_name}")

    raw_dataset = raw_dataset.map(
        lambda x: build_inputs_batch(x, tokenizer),
        batched=True,
        batch_size=512,
        num_proc=os.cpu_count(),
        desc="Pre-tokenizing",
    )

    raw_dataset = raw_dataset.filter(
        lambda x: x["len_generated"] <= MAX_INPUT_TOKENS,
        num_proc=os.cpu_count(),
    )

    dataloader = DataLoader(
        raw_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=streaming_collate_fn,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    hidden_dim = activation_generator.model.config.hidden_size
    writer = FastH5WriterCompress(
        path=os.path.join(OUTPUT_DIR, f"{split_name}.h5"),
        total_samples=len(raw_dataset),
        seq_dim=MAX_SEQ_LEN,
        hidden_dim=hidden_dim,
    )

    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Split: {split_name}"):
        tok_prompts = {
            k: v.to(config["llm_device"]) for k, v in batch["tokenized_prompts"].items()
        }
        tok_preferred = {
            k: v.to(config["llm_device"])
            for k, v in batch["tokenized_preferred"].items()
        }
        tok_rejected = {
            k: v.to(config["llm_device"])
            for k, v in batch["tokenized_rejected"].items()
        }

        with torch.no_grad():
            activations_masks = activation_generator.extract_activations(
                [
                    {
                        "input_ids": tok_prompts["input_ids"],
                        "attention_mask": tok_prompts["attention_mask"],
                        "prompts": batch["prompts_generated"],
                    }
                ],
                [
                    {
                        "input_ids": tok_preferred["input_ids"],
                        "attention_mask": tok_preferred["attention_mask"],
                        "prompts": batch["prompts_preferred"],
                    }
                ],
                [
                    {
                        "input_ids": tok_rejected["input_ids"],
                        "attention_mask": tok_rejected["attention_mask"],
                        "prompts": batch["prompts_rejected"],
                    }
                ],
            )

        gen_activations, gen_masks, gen_responses = activations_masks["generated"]
        pref_activations, pref_masks = activations_masks["preferred"]
        rej_activations, rej_masks = activations_masks["rejected"]

        gen_activations = pad_tensor(gen_activations, MAX_SEQ_LEN)
        pref_activations = pad_tensor(pref_activations, MAX_SEQ_LEN)
        rej_activations = pad_tensor(rej_activations, MAX_SEQ_LEN)

        gen_masks = pad_mask(gen_masks, MAX_SEQ_LEN)
        pref_masks = pad_mask(pref_masks, MAX_SEQ_LEN)
        rej_masks = pad_mask(rej_masks, MAX_SEQ_LEN)

        batch_dict = {
            "generated_activations": gen_activations.to(torch.float16).cpu().numpy(),
            "generated_masks": gen_masks.to(torch.uint8).cpu().numpy(),
            "generated_responses": [r["response"] for r in gen_responses],
            "preferred_activations": pref_activations.to(torch.float16).cpu().numpy(),
            "preferred_masks": pref_masks.to(torch.uint8).cpu().numpy(),
            "rejected_activations": rej_activations.to(torch.float16).cpu().numpy(),
            "rejected_masks": rej_masks.to(torch.uint8).cpu().numpy(),
        }

        writer.write_batch(batch_dict)

    writer.close()
    print(f"Finished writing {split_name}.h5")


def main():
    config = load_config(CONFIG_PATH)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    llm = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    activation_generator = ActivationsGenerator(
        llm, tokenizer, f"cuda:{config['llm_device']}"
    )

    dataset = load_dataset("stanfordnlp/SHP", split="train")

    process_split("train", dataset, activation_generator, tokenizer, config)


if __name__ == "__main__":
    main()
