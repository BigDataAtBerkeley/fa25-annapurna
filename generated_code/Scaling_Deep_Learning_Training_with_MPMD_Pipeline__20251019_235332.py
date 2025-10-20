import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Dataset Acquisition
def download_dataset(dataset_name='mnist', data_dir='data'):
    import os
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError(f'Dataset {dataset_name} is not supported.')

    return train_dataset, test_dataset

# Data Loading and Preprocessing
def get_data_loaders(dataset_name='mnist', batch_size=64, data_dir='data'):
    train_dataset, test_dataset = download_dataset(dataset_name, data_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Model Architecture
class PipelineParallelModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_stages):
        super(PipelineParallelModel, self).__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList([nn.Linear(input_dim, hidden_dim), *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_stages - 2)], nn.Linear(hidden_dim, output_dim)])

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
            x = F.relu(x)
        return x

# Training Loop
def train(model, train_loader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

# Evaluation
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Visualization
def visualize_results(model, test_loader, device, num_images=16):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray_r')
            ax.set_title(f'Label: {labels[i]}\nPrediction: {predicted[i]}')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.show()

# Usage Example
if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders
    train_loader, test_loader = get_data_loaders('mnist', batch_size=64)

    # Initialize model
    input_dim = 28 * 28
    hidden_dim = 128
    output_dim = 10
    num_stages = 4
    model = PipelineParallelModel(input_dim, hidden_dim, output_dim, num_stages).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    train(model, train_loader, optimizer, criterion, device, epochs=10)

    # Evaluate the model
    test(model, test_loader, device)

    # Visualize results
    visualize_results(model, test_loader, device)