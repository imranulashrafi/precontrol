import os
import yaml

with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["HF_HOME"] = config["hf_home"]

import torch
from torch.utils.data import DataLoader
from src.models.reward_model.factory_reward_model import RewardModel
from src.data.dataset_h5 import H5BatchDataset
from tqdm import tqdm

h5_path = "/home/iashrafi/Data/Codes/refactored_final/dataset/shp/train_shp.h5"
reward_model_name = "openbmb/UltraRM-13b"

dataset = H5BatchDataset(h5_path, batch_size=16)
dataloader = DataLoader(dataset, batch_size=None, shuffle=False)

reward_model = RewardModel(reward_model_name)

all_rewards = []

for batch in tqdm(dataloader, desc="Scoring responses"):
    responses = batch["generated_responses"]
    delimeter = "\nAssistant: "

    for response in responses:
        try:
            reward_output = reward_model.score(
                response.split(delimeter)[0] + delimeter, response.split(delimeter)[1]
            )
        except:
            print(response)
        all_rewards.append([reward_output.item()])

reward_tensor = torch.tensor(all_rewards, dtype=torch.float32)
print(f"Reward tensor shape: {reward_tensor.shape}")

torch.save(
    reward_tensor,
    "/home/iashrafi/Data/Codes/refactored_final/dataset/shp/rewards_ultrarm_train_shp.pt",
)
