import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from dataset_loader import load_dataset

class RingmasterASGD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RingmasterASGD, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load CIFAR-10 dataset
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# Initialize model, loss, and optimizer
input_dim = 3 * 32 * 32  # CIFAR-10 images are 3x32x32
hidden_dim = 512
output_dim = 10  # CIFAR-10 has 10 classes
model = RingmasterASGD(input_dim, hidden_dim, output_dim)

# Get Trainium device via Neuron SDK
device = xm.xla_device()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.view(inputs.size(0), -1).to(device)  # Flatten and move to device
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Use XLA-compatible optimizer step
        xm.optimizer_step(optimizer)
        # Synchronize XLA computation
        xm.mark_step()
        
        running_loss += loss.item()
        
        if i % 100 == 99:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.view(inputs.size(0), -1).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')

# Save the model
torch.save(model.state_dict(), 'ringmaster_asgd_model.pth')