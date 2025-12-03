# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from dataset_loader import load_dataset

# Model Definition
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dataset Loading
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# Instantiate Model
model = ConvNet()

# REQUIRED: Get Trainium device via Neuron SDK
device = xm.xla_device()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training Loop
for epoch in range(5):
    epoch_loss = 0.0
    num_batches = 0
    model.train()

    for batch_data in train_loader:
        inputs, labels = batch_data

        # Move tensors to device BEFORE operations
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # REQUIRED: Use Neuron SDK XLA optimizer step
        xm.optimizer_step(optimizer)

        # REQUIRED: Synchronize XLA computation
        xm.mark_step()

        # CRITICAL: loss.item() after xm.mark_step() for synchronization
        epoch_loss += float(loss.item())
        num_batches += 1

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    print(f"Epoch {epoch+1}/5, Average Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
test_loss = 0.0
correct = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += float(loss.item())

        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()

test_loss /= len(test_loader)
test_acc = correct / len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")