import yaml
import os

with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["HF_HOME"] = config["hf_home"]

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import wandb

from src.models.reward_model.factory_reward_model import RewardModel
from src.models.value_model.modeling_base import ValueFunctionModule
from src.models.value_model.modeling_margin import ValueFunctionModuleMargin
from src.data.dataset_h5 import H5BatchDataset
from transformers import AutoTokenizer
from torch.utils.data import Subset

pl.seed_everything(42)

wandb_logger = WandbLogger(
    project=config.get("wandb_project", "default_project"),
    name=config.get("wandb_run_name", "value_function_run"),
    config=config,
)

train_dataset = H5BatchDataset(
    config["h5_path_train"],
    batch_size=config["batch_size"],
    reward_tensor_path=config["reward_tensor_path_train"],
)

# train_dataset = Subset(train_dataset, indices=list(range(10)))

val_dataset = H5BatchDataset(
    config["h5_path_validation"],
    batch_size=config["batch_size"],
    reward_tensor_path=config["reward_tensor_path_validation"],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=None,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=None,
    shuffle=False,
)

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3", padding_side="left")

if config["experiment_name"] == "base":
    model = ValueFunctionModule(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        lr=config["lr"],
        tokenizer=tokenizer,
    )
elif config["experiment_name"] == "margin":
    model = ValueFunctionModuleMargin(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        lr=config["lr"],
        tokenizer=tokenizer,
    )

checkpoint_callback = ModelCheckpoint(
    dirpath=config.get("checkpoint_dir"),
    filename="epoch_{epoch}",
    save_top_k=3,
    monitor="val_total_loss",
    mode="min",
    save_last=True,
    every_n_epochs=1,
)

early_stop_callback = EarlyStopping(
    monitor="val_total_loss",
    patience=3,
    mode="min",
    verbose=True,
)

trainer = pl.Trainer(
    max_epochs=config["epochs"],
    devices=[config["model_device"]],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=10,
    num_sanity_val_steps=1,
    callbacks=[checkpoint_callback, early_stop_callback],
)

trainer.fit(model, train_loader, val_loader)
