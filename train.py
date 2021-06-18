from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in train_dataloader:
        batch, labels = batch.to(device), labels.to(device)
        logits = model(batch).logits
        print(labels.long())
        loss = lossfn(logits, labels.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_labels = torch.argmax(logits, axis=1)
        train_acc += (pred_labels == labels).sum().item()
        print("batch")
    train_loss /= len(train_dataset)
    train_acc /= len(train_dataset)
    return train_loss, train_acc


def train_model(num_epochs=1, batch_size=2):
    for epoch in range(1, num_epochs + 1):
        train_epoch(batch_size=batch_size)
        print(f"Finished epoch {epoch}")
