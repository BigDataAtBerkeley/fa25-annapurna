# Imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from dataset_loader import load_dataset
from transformers import AutoTokenizer

# Model definition
class AdamCBNet(nn.Module):
    def __init__(self):
        super(AdamCBNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Dataset loading
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# XLA device
device = xm.xla_device()
model = AdamCBNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 5
for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss.to(device)  # Move loss to XLA device
        loss.backward()
        
        # XLA optimizer step
        xm.optimizer_step(optimizer)
        xm.mark_step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += int(labels.size(0))  # Convert tensor size to Python int
        correct += (predicted == labels).sum().item()

print(f"Test accuracy: {100 * correct / total:.2f}%")