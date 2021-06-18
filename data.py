from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

checkpoint = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

full_dataset = load_dataset("sst", "default")


class SSTDataset(Dataset):
    def __init__(self, full_dataset, device, size=60, split="train"):
        self.raw_dataset = full_dataset[split]
        self.dataset = [
            (
                self.pad([tokenizer.cls_token_id] + tokenizer.encode(item["sentence"]) +
                         [tokenizer.sep_token_id], size=size),
                item["label"],
            )
            for item in self.raw_dataset
        ]
        self.device = device

    def pad(self, text, size=52):
        text_len = len(text)
        if text_len >= size:
            return text[:size]
        else:
            extra = size - len(text)
            return text + [0] * extra

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text, label = self.dataset[index]
        text = torch.tensor(text, device=self.device)
        label = torch.tensor(label, device=self.device)
        return text, label
