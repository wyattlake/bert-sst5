from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data import SSTDataset
import neptune.new as neptune
from tqdm import tqdm
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
eval_dataset = SSTDataset(full_dataset, device, split="validation")
test_dataset = SSTDataset(full_dataset, device, split="test")


# Neptune
run = neptune.init(project='wyattlake/bert-sst5',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwM2E2YmY0Yy02OTgxLTQ0NmQtOTFiMi03NzllNzk3OWM1YzAifQ==')
run["parameters"] = {"model": checkpoint, "learning_rate": 1e-5,
                     "optimizer": optimizer, "batch_size": 32, "epochs": 30}

run["size/train"] = len(train_dataset)
run["size/val"] = len(eval_dataset)
run["size/test"] = len(test_dataset)


def train_epoch(batch_size, dataset):
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in tqdm(train_dataloader):
        optimizer.zero_grad()
        batch, labels = batch.to(device), labels.to(device)
        logits = model(batch).logits
        loss = lossfn(logits, labels.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_labels = torch.argmax(logits, axis=1)
        train_acc += (pred_labels == labels).sum().item()

    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc


def eval_epoch(batch_size, dataset):
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )
    model.eval()
    eval_loss, eval_acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(train_dataloader):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch).logits
            err = lossfn(logits, labels.long())
            eval_loss += err.item()
            pred_labels = torch.argmax(logits, axis=1)
            eval_acc += (pred_labels == labels).sum().item()

    eval_loss /= len(dataset)
    eval_acc /= len(dataset)
    return eval_loss, eval_acc


def train_model(num_epochs=30, batch_size=32, save=True):
    for epoch in range(1, num_epochs + 1):
        print(f"STARTING EPOCH {epoch}")
        train_loss, train_acc = train_epoch(
            batch_size=batch_size, dataset=train_dataset)
        eval_loss, eval_acc = eval_epoch(
            batch_size=batch_size, dataset=eval_dataset)
        print(
            f"FINISHED EPOCH {epoch}\n\nTraining\nLoss: {train_loss:.4f}\nAccuracy: {train_acc:.4f}\n\nEvaluation\nLoss: {eval_loss:.4f}\nAccuracy: {eval_acc:.4f}\n")

        run["train/loss"].log(train_loss)
        run["val/loss"].log(eval_loss)

        run["train/acc"].log(train_acc)
        run["val/acc"].log(eval_acc)

        if save:
            with open(f'results/results_{checkpoint}_{num_epochs}_epochs.txt', 'a+') as f:
                f.write(
                    f"\nFINISHED EPOCH {epoch}\n\nTraining\nLoss: {train_loss:.4f}\nAccuracy: {train_acc:.4f}\n\nEvaluation\nLoss: {eval_loss:.4f}\nAccuracy: {eval_acc:.4f}\n")
            with open(f'models/{checkpoint}/epoch_{epoch}.pt', 'w+') as f:
                torch.save(model, f'models/{checkpoint}/epoch_{epoch}.pt')
