import os
import yaml


with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["HF_HOME"] = config["hf_home"]

import torch

from transformers import AutoTokenizer
from src.models.value_model.modeling_margin_regularizer import (
    ValueFunctionModuleMarginRegularizer,
)
from src.models.value_model.modeling_base import (
    ValueFunctionModule,
)
from src.models.value_model.modeling_margin import (
    ValueFunctionModuleMargin,
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
elif config["experiment_name"] == "margin_regularizer":
    model = ValueFunctionModuleMarginRegularizer(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        lr=config["lr"],
        tokenizer=tokenizer,
    )

ckpt_path = os.path.join(config["checkpoint_dir"], "best_epoch.pth")
state_dict = torch.load(ckpt_path, map_location=torch.device(config["model_device"]))


model.load_state_dict(state_dict["state_dict"])

missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
assert missing != True and unexpected != True

model = model.to(config["model_device"])

# model.inference()
model.calculate_win_rate()
model.calculate_avg_reward()
model.calculate_logprob_delta()
