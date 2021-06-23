from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from dotenv import load_dotenv
from data import SSTDataset
import neptune.new as neptune
from tqdm import tqdm
import hydra
import torch
import os

# .env variables
load_dotenv()
NEPTUNE_TOKEN = os.environ.get("NEPTUNE_TOKEN")

# Cuda setup
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Neptune
run = neptune.init(project='wyattlake/bert-sst5',
                   api_token=NEPTUNE_TOKEN)

# Dataset loading
full_dataset = load_dataset("sst", "default")
train_dataset = SSTDataset(full_dataset, device, split="train")
eval_dataset = SSTDataset(full_dataset, device, split="validation")
test_dataset = SSTDataset(full_dataset, device, split="test")


def train_epoch(model, batch_size, dataset, lossfn, optimizer, accumulation_steps):
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )
    model.train()
    optimizer.zero_grad()
    train_loss, train_acc, step = 0.0, 0.0, 0
    for batch, labels in tqdm(train_dataloader):
        batch, labels = batch.to(device), labels.to(device)
        output = model(batch)
        logits = output[0].squeeze()
        loss = lossfn(logits, labels.long())
        loss.backward()

        # Calculating loss and accuracy
        train_loss += loss.item()
        pred_labels = torch.argmax(logits, axis=1)
        batch_acc = (pred_labels == labels).sum().item()
        train_acc += batch_acc

        # Gradient accumulation
        step += 1
        if step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Logging values with Neptune
        run["train/loss"].log(loss.item() / batch_size)
        run["train/acc"].log(batch_acc / batch_size)

    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc


def eval_epoch(model, batch_size, dataset, lossfn):
    eval_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
    )
    model.eval()
    eval_loss, eval_acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(eval_dataloader):
            batch, labels = batch.to(device), labels.to(device)
            output = model(batch)
            logits = output[0].squeeze()
            err = lossfn(logits, labels.long())
            eval_loss += err.item()
            pred_labels = torch.argmax(logits, axis=1)
            eval_acc += (pred_labels == labels).sum().item()

    eval_loss /= len(dataset)
    eval_acc /= len(dataset)
    return eval_loss, eval_acc


@hydra.main(config_path="config", config_name="config")
def train_model(cfg, save=False):

    # Model settings
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.checkpoint, num_labels=5)
    lossfn = torch.nn.CrossEntropyLoss()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Neptune parameters
    run["parameters"] = {"model": cfg.checkpoint, "learning_rate": cfg.learning_rate,
                         "optimizer": optimizer, "batch_size": 32, "epochs": cfg.epochs}

    run["size/train"] = len(train_dataset)
    run["size/val"] = len(eval_dataset)
    run["size/test"] = len(test_dataset)

    for epoch in range(1, cfg.epochs + 1):
        print(f"STARTING EPOCH {epoch}")
        train_loss, train_acc = train_epoch(
            model, batch_size=cfg.batch.train_size, dataset=train_dataset, lossfn=lossfn, optimizer=optimizer, accumulation_steps=cfg.batch.accumulation_steps)
        eval_loss, eval_acc = eval_epoch(
            model, batch_size=cfg.batch.eval_size, dataset=eval_dataset, lossfn=lossfn)
        print(
            f"FINISHED EPOCH {epoch}\n\nTraining\nLoss: {train_loss:.4f}\nAccuracy: {train_acc:.4f}\n\nEvaluation\nLoss: {eval_loss:.4f}\nAccuracy: {eval_acc:.4f}\n")

        run["train_epoch/loss"].log(train_loss)
        run["val_epoch/loss"].log(eval_loss)

        run["train_epoch/acc"].log(train_acc)
        run["val_epoch/acc"].log(eval_acc)

        if save:
            with open(f'results/results_{cfg.checkpoint}_{cfg.epochs}_epochs.txt', 'a+') as f:
                f.write(
                    f"\nFINISHED EPOCH {epoch}\n\nTraining\nLoss: {train_loss:.4f}\nAccuracy: {train_acc:.4f}\n\nEvaluation\nLoss: {eval_loss:.4f}\nAccuracy: {eval_acc:.4f}\n")
            with open(f'models/{cfg.checkpoint}/epoch_{epoch}.pt', 'w+') as f:
                torch.save(
                    model, f'models/{cfg.checkpoint}/epoch_{cfg.epochs}.pt')
