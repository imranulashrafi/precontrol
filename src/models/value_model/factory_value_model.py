import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml
from tqdm import tqdm
from torch.utils.data import Subset
import copy
import os
import json

from torch.utils.data import DataLoader
from src.utils.inference_intervention import InferenceIntervention
from src.evaluation.win_rate import WinRateEvaluator
from src.data.dataset_shp import SHPDataset
from src.data.dataset_hhrlhf import HHRLHFDataset
from abc import ABC, abstractmethod
from src.models.reward_model.factory_reward_model import RewardModel


with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)


class ValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class BaseValueFunctionModule(pl.LightningModule, ABC):
    def __init__(self, input_dim, hidden_dim, output_dim, lr, tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=["reward_scorer", "tokenizer"])
        self.model = ValueFunction(input_dim, hidden_dim, output_dim)
        self.lr = lr
        self.tokenizer = tokenizer

    def forward(self, x):
        return self.model(x)

    def inference(self):
        if config["dataset_name"] == "shp":
            dataset = SHPDataset(config["test_data_path"])
        if config["dataset_name"] == "hhrlhf":
            dataset = HHRLHFDataset(config["test_data_path"])
            dataset = Subset(dataset, indices=list(range(1000)))

        prompts = [d["prompt"] for d in dataset]
        preferreds = [d["preferred"] for d in dataset]
        rejecteds = [d["rejected"] for d in dataset]

        encodings = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        input_dicts = [
            {k: v[i] for k, v in encodings.items()} for i in range(len(prompts))
        ]

        batch_size = 8
        dataloader = DataLoader(
            input_dicts,
            batch_size=batch_size,
            collate_fn=lambda x: self.tokenizer.pad(x, return_tensors="pt"),
        )

        value_model = copy.deepcopy(self.model).to(self.device)

        infer = InferenceIntervention(
            model_path=config["model_name"],
            tokenizer=self.tokenizer,
            value_model=value_model,
            use_intervention=True,
        )

        generated_outputs = []
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Inference Intervention", total=len(dataloader))
        ):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch["input_ids"].shape[0]
            prompt_batch = prompts[start_idx:end_idx]

            results = infer.generate_and_decode(batch, prompt_batch)
            generated_outputs.extend(results)

        generations = [item["result"] for item in generated_outputs]

        for prompt, preferred, rejected, generation in zip(
            prompts, preferreds, rejecteds, generations
        ):
            entry = {
                "prompt": prompt,
                "preferred": preferred,
                "rejected": rejected,
                "generation": generation,
            }
            results.append(entry)

        output_path = os.path.join(config["checkpoint_dir"], "test_results")
        os.makedirs(output_path, exist_ok=True)

        with open(
            os.path.join(output_path, "inference_results.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(results)} entries to {output_path}")

    def calculate_win_rate(self):
        if not os.path.exists(os.path.join(config["checkpoint_dir"], "test_results")):
            self.inference()

        with open(
            os.path.join(
                config["checkpoint_dir"], "test_results", "inference_results.json"
            ),
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        prompts = [item["prompt"] for item in data]
        preferreds = [item["preferred"] for item in data]
        generations = [item["generation"] for item in data]

        evaluator = WinRateEvaluator()
        win_count = 0
        total = 0
        for prompt, gen, pref in tqdm(
            zip(prompts, generations, preferreds),
            total=len(prompts),
            desc="Evaluating Win Rate",
        ):
            pref = pref.rsplit("Assistant: ", 1)
            pref = pref[1] if len(pref) > 1 else ""
            
            score_gen, score_pref, _ = evaluator.evaluate_pair(prompt, gen, pref)
            if score_gen > score_pref:
                win_count += 1
            total += 1

        win_rate = (win_count / total) * 100 if total > 0 else 0.0

        print(f"📊 Win Rate: {win_rate:.2f}%")

    def calculate_avg_reward(self):
        if not os.path.exists(os.path.join(config["checkpoint_dir"], "test_results")):
            self.inference()

        with open(
            os.path.join(
                config["checkpoint_dir"], "test_results", "inference_results.json"
            ),
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        prompts = [item["prompt"] for item in data]
        generations = [item["generation"] for item in data]

        reward_model = RewardModel("openbmb/UltraRM-13b")

        rewards = []

        for prompt, generation in tqdm(zip(prompts, generations)):
            rewards.append(reward_model.score(prompt, generation).item())

        print(f"📊 Avg. Reward: {sum(rewards)/len(rewards)}")

    @abstractmethod
    def shared_step(self, batch, stage):
        pass

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
