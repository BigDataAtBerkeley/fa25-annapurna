import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from dataset_loader import load_dataset

# LoRA Linear Layer
class LoRALinear(nn.Module):
    def __init__(self, layer, rank=4):
        super().__init__()
        self.layer = layer  # Original layer (e.g., nn.Linear)
        # A: [rank, in_features], B: [out_features, rank]
        self.A = nn.Parameter(torch.zeros(rank, layer.in_features))
        self.B = nn.Parameter(torch.zeros(layer.out_features, rank))
        # Initialize properly
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        # CORRECT: W' = W + B@A, then y = x @ W'^T + b
        return F.linear(x, self.layer.weight + self.B @ self.A, self.layer.bias)

# LoRA Model Wrapper
class LoRAModel(nn.Module):
    def __init__(self, model, rank=4):
        super().__init__()
        self.layers = nn.ModuleList()
        # Wrap nn.Sequential models with LoRA layers
        for layer in model:
            if isinstance(layer, nn.Linear):
                self.layers.append(LoRALinear(layer, rank))
            else:
                self.layers.append(layer)  # Keep non-linear layers as-is

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Load dataset
train_loader, test_loader = load_dataset('mnist', batch_size=128)

# Define base model
base_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Wrap base model with LoRA
model = LoRAModel(base_model, rank=4)

# Move model to XLA device
device = xm.xla_device()
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        # Move tensors to XLA device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)  # Neuron SDK XLA optimizer step
        xm.mark_step()  # Synchronize XLA computation

    # Evaluation
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
    print(f'Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.2f}%')