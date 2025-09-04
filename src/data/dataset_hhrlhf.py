import json
from torch.utils.data import Dataset


class HHRLHFDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.dataset = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print(item)

        prompt = (
            item.get("chosen")
            .rsplit("Assistant:", 1)[0]
            .strip()
            .replace("Human:", "User:")
            + "\n\nAssistant:"
        )
        preferred = item.get("chosen").strip().replace("Human:", "User:")
        rejected = item.get("rejected").strip().replace("Human:", "User:")

        return {"prompt": prompt, "preferred": preferred, "rejected": rejected}
