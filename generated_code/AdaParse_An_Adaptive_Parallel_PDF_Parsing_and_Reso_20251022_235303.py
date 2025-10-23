import os
import shutil
import requests
from tqdm import tqdm
import pdfplumber
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# Dataset Acquisition
def download_dataset(dataset_url: str, dataset_path: str = 'data'):
    """
    Download the dataset from the provided URL and save it to the specified path.
    If the dataset already exists locally, it will not be downloaded again.
    """
    os.makedirs(dataset_path, exist_ok=True)
    filename = dataset_url.split('/')[-1]
    filepath = os.path.join(dataset_path, filename)

    if not os.path.exists(filepath):
        print(f"Downloading dataset from {dataset_url}")
        response = requests.get(dataset_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(filepath, 'wb') as file:
            for data in response.iter_content(chunk_size=block_size):
                if data:
                    file.write(data)
                    progress_bar.update(len(data))

        progress_bar.close()
        print(f"Dataset downloaded and saved to {filepath}")
    else:
        print(f"Dataset already exists at {filepath}")

# Data Loading and Preprocessing
def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file using pdfplumber library.
    """
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def preprocess_data(dataset_path):
    """
    Preprocess the dataset by extracting text from PDF files.
    """
    data = []
    for file in os.listdir(dataset_path):
        if file.endswith('.pdf'):
            pdf_file = os.path.join(dataset_path, file)
            text = extract_text_from_pdf(pdf_file)
            data.append(text)
    return data

# Model Architecture
class AdaptiveParserModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdaptiveParserModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_text):
        embedded = self.embedding(input_text)
        output, _ = self.lstm(embedded)
        output = self.fc(output[:, -1, :])
        return output

# Training Loop
def train(model, data_loader, optimizer, criterion, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Evaluation
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# Visualization
def plot_accuracy(accuracies, title):
    plt.figure(figsize=(8, 6))
    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.show()

# Usage Example
if __name__ == "__main__":
    # Dataset Acquisition
    dataset_url = "https://example.com/dataset.zip"
    download_dataset(dataset_url)

    # Data Loading and Preprocessing
    dataset_path = "data"
    data = preprocess_data(dataset_path)

    # Split data into train and test sets
    train_data, test_data = torch.utils.data.random_split(data, [0.8, 0.2])

    # Create data loaders
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model Architecture
    input_size = len(train_data.vocab)
    hidden_size = 256
    output_size = 2  # Binary classification (parse or not parse)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveParserModel(input_size, hidden_size, output_size).to(device)

    # Training Loop
    num_epochs = 10
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    train(model, train_loader, optimizer, criterion, device, num_epochs)

    # Evaluation
    test_accuracy = evaluate(model, test_loader, device)

    # Visualization
    accuracies = [test_accuracy]
    plot_accuracy(accuracies, "Test Accuracy")