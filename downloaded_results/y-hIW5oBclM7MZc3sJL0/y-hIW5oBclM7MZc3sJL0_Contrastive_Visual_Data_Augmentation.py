import math
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer
from dataset_loader import load_dataset

device = xm.xla_device()

class ContrastiveVisualDataAugmentation(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, hidden_dim=256):
        super(ContrastiveVisualDataAugmentation, self).__init__()
        self.embedding = nn.Linear(3 * 32 * 32, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.encoder(x)
        return self.classifier(x)

# Load dataset
train_loader, test_loader = load_dataset('cifar100', batch_size=128)

# Initialize model, loss, and optimizer
model = ContrastiveVisualDataAugmentation(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):
    epoch_loss = 0.0
    num_batches = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        xm.optimizer_step(optimizer)
        xm.mark_step()
        
        epoch_loss += float(loss.item())
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    print(f"Epoch {epoch+1}/5, Average Loss: {avg_loss:.4f}")

# Evaluation on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += int(labels.size(0))
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")