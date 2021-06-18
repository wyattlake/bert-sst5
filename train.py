from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from data import SSTDataset
import torch

# Cuda setup
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Model settings
checkpoint = "bert-large-uncased"
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
lossfn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Dataset loading
full_dataset = load_dataset("sst", "default")
train_dataset = SSTDataset(full_dataset, device, split="train")


def train_epoch(batch_size):
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size,
    )
    model.train()
    for batch, labels in train_dataloader:
        break


def train_model(num_epochs=1, batch_size=2):
    for epoch in range(1, num_epochs + 1):
        train_epoch(batch_size=batch_size)
        print(f"Finished epoch {epoch}")
