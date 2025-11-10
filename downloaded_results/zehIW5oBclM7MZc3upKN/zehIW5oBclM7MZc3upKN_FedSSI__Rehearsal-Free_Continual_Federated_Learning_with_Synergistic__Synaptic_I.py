import torch
import torch.nn as nn
from dataset_loader import load_dataset

# XLA-compatible sigmoid implementation
def xla_sigmoid(x):
    return torch.reciprocal(1 + torch.exp(-x))

# FedSSI Model
class FedSSIModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512, 256], dropout=0.2):
        super(FedSSIModel, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        
        layers = []
        prev_size = input_size
        for i, size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(self.dropout)
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.layers = nn.Sequential(*layers)
        
        self.si_weights = nn.ParameterList([nn.Parameter(torch.ones(size)) for size in hidden_sizes + [output_size]])
        self.si_biases = nn.ParameterList([nn.Parameter(torch.zeros(size)) for size in hidden_sizes + [output_size]])
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.layers(x)
        si_weights = [weight * xla_sigmoid(bias) for weight, bias in zip(self.si_weights, self.si_biases)]
        out_chunks = torch.split(out, list(out.shape)[1] // (self.num_layers + 1), dim=1)
        out = [out_layer * si_weight for out_layer, si_weight in zip(out_chunks, si_weights)]
        out = torch.cat(out, dim=1)
        return out

# Dataset Loading
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# Training Loop
import torch_xla.core.xla_model as xm
device = xm.xla_device()
model = FedSSIModel(input_size=3072, output_size=10, hidden_sizes=[512, 256]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(5):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_data in train_loader:
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        xm.mark_step()
        loss.backward()
        xm.optimizer_step(optimizer)
        
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
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")