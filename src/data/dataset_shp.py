import json
from torch.utils.data import Dataset


class SHPDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.dataset = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        prompt = "User: " + item.get("history", "") + "\nAssistant:"

        label = item.get("labels")
        comment_a = item.get("human_ref_A", "")
        comment_b = item.get("human_ref_B", "")

        if label == 1:
            preferred = comment_a
            rejected = comment_b
        else:
            preferred = comment_b
            rejected = comment_a

        return {"prompt": prompt, "preferred": preferred, "rejected": rejected}
