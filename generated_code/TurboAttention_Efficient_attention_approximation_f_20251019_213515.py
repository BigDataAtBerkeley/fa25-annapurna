import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Synthetic Data Generator
def generate_synthetic_data(num_samples, input_dim, output_dim):
    """Generate synthetic data with similar characteristics as the paper's dataset."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))
    return X, y

# Dataset Preparation
def prepare_dataset(num_samples=10000, input_dim=1024, output_dim=10):
    """Prepare a synthetic dataset and save it to the 'data/' directory."""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    X, y = generate_synthetic_data(num_samples, input_dim, output_dim)
    torch.save((X, y), os.path.join(data_dir, 'synthetic_dataset.pt'))
    print(f'Synthetic dataset saved to {data_dir}/synthetic_dataset.pt')

# Data Loading and Preprocessing
def load_dataset(batch_size=128):
    """Load the synthetic dataset and create data loaders."""
    data_dir = 'data'
    dataset_path = os.path.join(data_dir, 'synthetic_dataset.pt')

    if not os.path.exists(dataset_path):
        print('Dataset not found. Generating synthetic dataset...')
        prepare_dataset()

    X, y = torch.load(dataset_path)
    dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# Model Architecture
class TurboAttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TurboAttentionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Training Loop
def train(model, train_loader, epochs=10, lr=0.001):
    """Train the TurboAttention model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

# Evaluation
def evaluate(model, test_loader):
    """Evaluate the TurboAttention model."""
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
    print(f'Accuracy: {accuracy:.2f}%')

# Visualization
def visualize_examples(model, test_loader):
    """Visualize some examples from the test dataset."""
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Plot the first 10 examples
            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            for i in range(10):
                ax = axs[i // 5, i % 5]
                ax.imshow(inputs[i].cpu().numpy().reshape(32, 32), cmap='gray')
                ax.set_title(f'Predicted: {predicted[i].item()}')
                ax.axis('off')
            plt.show()
            break

# Main Function
def main():
    # Prepare the dataset
    train_loader = load_dataset(batch_size=128)

    # Create the model
    model = TurboAttentionModel(input_dim=1024, output_dim=10).to(device)

    # Train the model
    train(model, train_loader, epochs=10, lr=0.001)

    # Evaluate the model
    evaluate(model, train_loader)

    # Visualize examples
    visualize_examples(model, train_loader)

if __name__ == '__main__':
    main()