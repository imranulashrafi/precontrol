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


class ValueFunctionModule(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        lr,
        tokenizer,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["reward_scorer", "tokenizer"])
        self.model = ValueFunction(input_dim, hidden_dim, output_dim)
        self.lr = lr

        self.tokenizer = tokenizer

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage):
        activations = batch["generated_activations"]
        masks = batch["generated_masks"]
        responses = batch["generated_responses"]
        rewards = batch["rewards"]

        batch_size, seq_len, hidden_dim = activations.shape
        predictions = self(activations.view(-1, hidden_dim)).view(
            batch_size, seq_len, -1
        )

        valid_mask = masks[:, :-1] * masks[:, 1:]
        valid_preds = predictions[:, :-1][valid_mask.bool()]
        next_valid_preds = predictions[:, 1:][valid_mask.bool()]
        pairwise_loss = F.mse_loss(valid_preds, next_valid_preds, reduction="sum")

        last_indices = masks.float().argmax(dim=1, keepdim=True)
        last_indices[masks.sum(dim=1) == 0] = -1
        batch_indices = torch.arange(batch_size, device=self.device)
        final_preds = predictions[batch_indices, last_indices.squeeze()]
        final_loss = F.mse_loss(final_preds, rewards, reduction="sum")

        total_loss = (pairwise_loss + final_loss) / batch_size

        is_train = stage == "train"
        is_val = stage == "val"

        self.log(
            f"{stage}_total_loss",
            total_loss,
            on_step=is_train,
            on_epoch=is_val,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_pairwise_loss",
            pairwise_loss / batch_size,
            on_step=is_train,
            on_epoch=is_val,
            logger=True,
        )
        self.log(
            f"{stage}_reward_loss",
            final_loss / batch_size,
            on_step=is_train,
            on_epoch=is_val,
            logger=True,
        )

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def on_validation_epoch_end(self):
        full_dataset = SHPDataset(config["validation_data_path"])
        dataset = Subset(full_dataset, indices=list(range(100)))

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

        total_logprob_delta = 0.0
        for prompt, pref, rej in tqdm(
            zip(prompts, preferreds, rejecteds),
            total=len(prompts),
            desc="Computing LogProb Delta",
        ):
            delta = compute_logprob_delta(
                prompt, pref, rej, infer.model, self.tokenizer
            )
            total_logprob_delta += delta

        avg_logprob_delta = total_logprob_delta / len(prompts)
        self.log("val_logprob_delta", avg_logprob_delta, on_epoch=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
