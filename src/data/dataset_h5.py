import h5py
import torch
from torch.utils.data import Dataset
import hdf5plugin
import os


class H5BatchDataset(Dataset):
    def __init__(self, h5_path, batch_size, reward_tensor_path=None):
        self.h5_file = h5py.File(h5_path, "r")
        self.batch_size = batch_size
        self.total_samples = self.h5_file["generated/activations"].shape[0]
        self.num_batches = self.total_samples // batch_size

        if reward_tensor_path is not None:
            self.rewards = torch.load(reward_tensor_path)
        else:
            self.rewards = None

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        s = idx * self.batch_size
        e = s + self.batch_size

        batch = {
            "generated_activations": torch.tensor(
                self.h5_file["generated/activations"][s:e], dtype=torch.float16
            ),
            "generated_masks": torch.tensor(
                self.h5_file["generated/masks"][s:e], dtype=torch.uint8
            ),
            "generated_responses": [
                x.decode("utf-8") for x in self.h5_file["generated/responses"][s:e]
            ],
            "preferred_activations": torch.tensor(
                self.h5_file["preferred/activations"][s:e], dtype=torch.float16
            ),
            "preferred_masks": torch.tensor(
                self.h5_file["preferred/masks"][s:e], dtype=torch.uint8
            ),
            "rejected_activations": torch.tensor(
                self.h5_file["rejected/activations"][s:e], dtype=torch.float16
            ),
            "rejected_masks": torch.tensor(
                self.h5_file["rejected/masks"][s:e], dtype=torch.uint8
            ),
        }

        if self.rewards is not None:
            batch["rewards"] = self.rewards[s:e]

        return batch
