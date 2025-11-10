import math
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
from dataset_loader import load_dataset

class NeuralGuidedDiffusionBridge(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralGuidedDiffusionBridge, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, test_loader, epochs, device):
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            xm.optimizer_step(optimizer)
            xm.mark_step()
            
            epoch_loss += loss.sum().item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            num_batches = 0
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.sum().item()
                num_batches += 1
            
            avg_test_loss = test_loss / num_batches if num_batches > 0 else 0.0
            print(f"Test Average Loss: {avg_test_loss:.4f}")
            
    return model

xm.rendezvous()  # Initialize XLA distributed environment

device = xm.xla_device()
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

input_size = 3 * 32 * 32  # CIFAR-10 image size (channels, height, width)
hidden_size = 128
output_size = 10  # Number of classes in CIFAR-10

model = NeuralGuidedDiffusionBridge(input_size, hidden_size, output_size).to(device)
model = train_model(model, train_loader, test_loader, epochs=10, device=device)

xm.save(model.state_dict(), "trained_model.pt")