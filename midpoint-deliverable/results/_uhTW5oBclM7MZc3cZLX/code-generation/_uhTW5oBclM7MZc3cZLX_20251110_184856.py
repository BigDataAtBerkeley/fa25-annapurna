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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Dataset Loading
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# Model Instantiation
model = ConvNet()

# Get XLA device
device = xm.xla_device()
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training Loop
for epoch in range(5):
    epoch_loss = 0.0
    num_batches = 0

    for batch_data in train_loader:
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # XLA Optimizer Step
        xm.optimizer_step(optimizer)

        # Synchronize XLA Computation
        xm.mark_step()

        epoch_loss += float(loss.item())
        num_batches += 1

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    print(f"Epoch {epoch+1}/5, Average Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_data in test_loader:
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += int(labels.size(0))
        correct += int((predicted == labels).sum().item())

print(f"Test Accuracy: {100 * correct / total:.2f}%")