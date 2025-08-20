import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml
from tqdm import tqdm
from torch.utils.data import Subset
import copy

from torch.utils.data import DataLoader
from src.utils.inference_intervention import InferenceIntervention
from src.evaluation.win_rate import WinRateEvaluator
from src.evaluation.logprob_delta import compute_logprob_delta
from src.data.dataset_shp import SHPDataset
from abc import ABC, abstractmethod

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
    
    def evaluation(self, split="valid"):
        if split == "valid":
            dataset = SHPDataset(config["validation_data_path"])
        elif split == "test":
            dataset = SHPDataset(config["test_data_path"])

        dataset = Subset(dataset, indices=list(range(1)))

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

        evaluator = WinRateEvaluator()
        win_count = 0
        total = 0
        for prompt, gen, pref in tqdm(
            zip(prompts, generations, preferreds),
            total=len(prompts),
            desc="Evaluating Win Rate",
        ):
            score_gen, score_pref, _ = evaluator.evaluate_pair(prompt, gen, pref)
            if score_gen > score_pref:
                win_count += 1
            total += 1

        win_rate = (win_count / total) * 100 if total > 0 else 0.0
        self.log("val_win_rate", win_rate, on_epoch=True, logger=True)

        macro_sum = 0
        correct = 0
        total = 0

        for prompt, pref, rej in zip(prompts, preferreds, rejecteds):
            delta = compute_logprob_delta(
                prompt, pref, rej, infer.model, self.tokenizer
            )
            macro_sum += delta
            correct += delta > 0.0
            total += 1

        macro_mean = macro_sum / total if total else 0.0
        acc = (correct / total) if total else 0.0

        self.log(
            "val_logprob_delta",
            macro_mean,
            on_epoch=True,
            logger=True,
        )
        self.log("val_logprob_delta_acc", acc, on_epoch=True, logger=True)

        if split=="test":
            print(f"📊 Win Rate: {win_rate:.2f}%")
            print(f"📊 Logprob Δ Mean: {macro_mean:.4f}")
            print(f"📊 Logprob Δ Accuracy: {acc * 100:.2f}%")

    @abstractmethod
    def shared_step(self, batch, stage):
        pass

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def on_validation_epoch_end(self):
        return self.evaluation()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
