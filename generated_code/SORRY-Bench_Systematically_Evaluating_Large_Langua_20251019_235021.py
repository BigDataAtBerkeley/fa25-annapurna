import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Dataset Acquisition
def download_dataset():
    if not os.path.exists('data/sorry_bench'):
        print("Downloading SORRY-Bench dataset...")
        dataset = load_dataset("sorry_bench", "sorry_bench")
        dataset.save_to_disk('data/sorry_bench')
        print("Dataset downloaded and saved to data/sorry_bench")
    else:
        print("SORRY-Bench dataset found locally.")

# Data Loading and Preprocessing
def load_data(batch_size=16):
    dataset = load_dataset("data/sorry_bench", "sorry_bench")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["prompt"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_loaders = {
        "train": DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True),
        "val": DataLoader(tokenized_datasets["val"], batch_size=batch_size),
        "test": DataLoader(tokenized_datasets["test"], batch_size=batch_size)
    }
    return data_loaders

# Model Architecture
class SafetyRefusalModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

# Training Loop
def train(model, data_loaders, num_epochs, lr=1e-5, device=torch.device('cpu')):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(data_loaders["train"], desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} loss: {total_loss / len(data_loaders['train'])}")

# Evaluation
def evaluate(model, data_loader, device=torch.device('cpu')):
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total_accuracy += (predictions == labels).sum().item()

    return total_accuracy / len(data_loader.dataset)

# Visualization
def plot_metrics(train_metrics, val_metrics, metric_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f"Train {metric_name}")
    plt.plot(val_metrics, label=f"Validation {metric_name}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} Curves")
    plt.legend()
    plt.show()

# Main Code
if __name__ == "__main__":
    download_dataset()
    data_loaders = load_data()

    model = SafetyRefusalModel()
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_metrics = []
    val_metrics = []
    for epoch in range(num_epochs):
        train(model, data_loaders, num_epochs=1, device=device)
        train_acc = evaluate(model, data_loaders["train"], device)
        val_acc = evaluate(model, data_loaders["val"], device)
        train_metrics.append(train_acc)
        val_metrics.append(val_acc)
        print(f"Epoch {epoch + 1} - Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

    plot_metrics(train_metrics, val_metrics, "accuracy")