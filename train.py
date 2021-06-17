from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cuda setup
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Model settings
checkpoint = "bert-large-uncased"
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


def train_model(num_epochs=10):
    loss = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, num_epochs + 1):
        print(f"Finished epoch {epoch}")
