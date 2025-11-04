# Datasets for Generated Code Testing

This directory contains scripts and utilities for managing datasets used by generated PyTorch code on Trainium instances.

## Overview

Generated code is tested on Trainium instances with standardized datasets stored in the S3 bucket **`datasets-for-all-papers`**. This ensures consistent, reproducible testing across all papers.

## Available Datasets

| Dataset | Type | Size | Samples | Use Case |
|---------|------|------|---------|----------|
| **CIFAR-10** | Image Classification | ~170 MB | 60K | Computer vision models, CNNs |
| **CIFAR-100** | Image Classification | ~170 MB | 60K | Multi-class vision tasks |
| **MNIST** | Image Classification | ~12 MB | 70K | Simple digit classification |
| **Fashion-MNIST** | Image Classification | ~30 MB | 70K | Fashion item classification |
| **IMDB** | Text Classification | ~65 MB | 50K | Sentiment analysis, NLP |
| **WikiText-2** | Language Modeling | ~12 MB | 36K | Language models, transformers |
| **Synthetic** | Various | ~500 MB | 16K | Quick testing, debugging |

## Setup Instructions

### 1. Upload Datasets to S3

Run the upload script to populate your S3 bucket:

```bash
cd /Users/danchizik/Desktop/annapurna/datasets

# Activate your virtual environment
source ../aws_env/bin/activate

# Install dependencies
pip install boto3 torch torchvision datasets tqdm

# Upload all datasets to S3
python upload_datasets_to_s3.py
```

**Expected output:**
```
Starting dataset upload to S3...
Target bucket: s3://datasets-for-all-papers
...
✓ All datasets uploaded successfully!
Total datasets: 7
Total size: 930.45 MB
```

**Time estimate:** 15-30 minutes depending on internet speed.

### 2. Verify Upload

Check that datasets are in S3:

```bash
# List all datasets
aws s3 ls s3://datasets-for-all-papers/

# Check master index
aws s3 cp s3://datasets-for-all-papers/dataset_index.json - | python -m json.tool
```

### 3. Deploy Dataset Loader to Trainium

The `dataset_loader.py` utility must be deployed to your Trainium instance:

```bash
cd /Users/danchizik/Desktop/annapurna/deployment

# Deploy to Trainium (this will include dataset_loader.py)
./deploy_trainium.sh /path/to/your-key.pem
```

Or manually:
```bash
# Copy dataset_loader.py to Trainium
scp -i your-key.pem \
  ../trainium_executor/dataset_loader.py \
  ubuntu@<TRAINIUM_IP>:~/trainium-executor/
```

## Usage in Generated Code

### Standard Pattern

Generated code should use the dataset loader utility:

```python
# Import the dataset loader (automatically available on Trainium)
from dataset_loader import load_dataset

# Load CIFAR-10 with custom batch size
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# Use in training loop
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your training code here
        pass
```

### Available Datasets

```python
from dataset_loader import list_available_datasets

# List all datasets
datasets = list_available_datasets()
print(datasets)
# Output: ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'imdb', 'wikitext2', 'synthetic']
```

### Dataset-Specific Examples

#### CIFAR-10 (Computer Vision)
```python
from dataset_loader import load_dataset
import torch.nn as nn
import torch.optim as optim

# Load data
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# Your model
model = YourCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

#### MNIST (Simple Vision)
```python
from dataset_loader import load_dataset

train_loader, test_loader = load_dataset('mnist', batch_size=64)
# Images are 28x28 grayscale
```

#### IMDB (NLP)
```python
from dataset_loader import load_dataset

train_data, test_data = load_dataset('imdb')
# Returns HuggingFace datasets with 'text' and 'label' fields
print(train_data[0])
# {'text': 'This movie was great!...', 'label': 1}
```

#### Synthetic (Quick Testing)
```python
from dataset_loader import load_dataset

# Load small synthetic dataset for quick testing
data = load_dataset('synthetic', variant='small')
print(data['description'])
# Small synthetic dataset: 1K samples, 224x224 RGB images, 10 classes

images = data['images']  # torch.Tensor (1000, 3, 224, 224)
labels = data['labels']  # torch.Tensor (1000,)
```

### Fallback Pattern

For robust code generation, use this fallback pattern:

```python
def load_data(batch_size=128):
    """Load dataset with fallback to synthetic data"""
    try:
        # Try loading from S3
        from dataset_loader import load_dataset
        train_loader, test_loader = load_dataset('cifar10', batch_size=batch_size)
        print("✓ Loaded CIFAR-10 from S3")
        return train_loader, test_loader
    except Exception as e:
        print(f"⚠ Failed to load from S3: {e}")
        print("Falling back to synthetic data generation...")
        
        # Fallback: generate synthetic data
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        X_train = torch.randn(5000, 3, 32, 32)
        y_train = torch.randint(0, 10, (5000,))
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        return train_loader, train_loader
```

## Updating Code Generation

To integrate datasets into your code generation pipeline:

### 1. Update Code Generation Prompt

In `code_gen/pytorch_generator.py`, modify the prompt to include dataset loading:

```python
prompt = f"""
Generate PyTorch code for this research paper.

IMPORTANT: Use the standardized dataset loader for data:

```python
from dataset_loader import load_dataset

# For vision tasks, use CIFAR-10:
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# For NLP tasks, use IMDB:
train_data, test_data = load_dataset('imdb')

# For quick testing, use synthetic data:
data = load_dataset('synthetic', variant='small')
```

Available datasets: cifar10, cifar100, mnist, fashion_mnist, imdb, wikitext2, synthetic

Paper: {paper_title}
...
"""
```

### 2. Example Generated Code Template

```python
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import load_dataset

def main():
    # Load dataset from S3 (cached on Trainium)
    print("Loading dataset...")
    train_loader, test_loader = load_dataset('cifar10', batch_size=128)
    print(f"✓ Loaded {len(train_loader)} training batches")
    
    # Model definition
    model = YourModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")
        
        # Output metrics for tracking
        print(f"METRICS: {{'epoch': {epoch+1}, 'training_loss': {avg_loss}}}")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"METRICS: {{'test_accuracy': {accuracy}}}")

if __name__ == '__main__':
    main()
```

## Testing Locally

Test the dataset loader before deploying to Trainium:

```bash
cd /Users/danchizik/Desktop/annapurna/trainium_executor

# Test dataset downloads
python dataset_loader.py

# Expected output:
# Available datasets:
#   - cifar10
#   - cifar100
#   - mnist
#   ...
# Testing CIFAR-10 download and load...
# ✓ CIFAR-10 loaded: 391 train batches, 79 test batches
```

## Cost Considerations

| Dataset | S3 Storage Cost/Month | Download Cost (per execution) |
|---------|----------------------|-------------------------------|
| CIFAR-10 | ~$0.004 | ~$0.017 (first time only) |
| IMDB | ~$0.002 | ~$0.007 |
| WikiText-2 | ~$0.0003 | ~$0.001 |
| **Total** | **~$0.025/month** | **Cache on Trainium = $0** |

**Note:** Datasets are cached on the Trainium instance at `/tmp/datasets`, so download costs are one-time per instance lifecycle.

## Monitoring

### Check Dataset Usage in OpenSearch

After tests complete, OpenSearch will include dataset metadata:

```json
{
  "paper_id": "abc123",
  "test_success": true,
  "dataset_name": "cifar10",
  "dataset_size_mb": 170.5,
  "dataset_download_time": 45.2,
  "execution_time": 320.5
}
```

### View Dataset Cache on Trainium

SSH into your Trainium instance:

```bash
ssh -i your-key.pem ubuntu@<TRAINIUM_IP>

# Check cached datasets
ls -lh /tmp/datasets/

# Output:
# drwxr-xr-x 3 ubuntu ubuntu 4.0K Oct 30 12:00 cifar10
# drwxr-xr-x 3 ubuntu ubuntu 4.0K Oct 30 12:05 mnist
# -rw-r--r-- 1 ubuntu ubuntu 2.1K Oct 30 12:00 dataset_index.json
```

## Troubleshooting

### Issue: "Dataset not found in S3"

**Solution:**
1. Verify bucket exists: `aws s3 ls s3://datasets-for-all-papers/`
2. Check IAM permissions on Trainium instance
3. Re-run upload script

### Issue: "Dataset loader import error"

**Solution:**
1. Ensure `dataset_loader.py` is in the Trainium executor directory
2. Redeploy using `./deploy_trainium.sh`

### Issue: "Dataset download too slow"

**Solution:**
1. Check Trainium instance is in same region as S3 bucket (should be us-east-1)
2. Use S3 VPC endpoint for faster transfers
3. Pre-warm cache by manually downloading datasets:
   ```bash
   ssh ubuntu@<TRAINIUM_IP>
   cd ~/trainium-executor
   python3 -c "from dataset_loader import download_dataset; download_dataset('cifar10')"
   ```

## Adding New Datasets

To add a custom dataset:

1. **Add download function** to `upload_datasets_to_s3.py`:
   ```python
   def download_my_dataset():
       dataset_dir = os.path.join(LOCAL_CACHE, 'my_dataset')
       # Download and prepare your dataset
       # ...
       upload_directory_to_s3(dataset_dir, 'my_dataset')
       create_dataset_metadata(...)
   ```

2. **Add loader function** to `trainium_executor/dataset_loader.py`:
   ```python
   def _load_my_dataset(self, dataset_dir: str, **kwargs):
       # Load your dataset
       return train_data, test_data
   ```

3. **Upload to S3**:
   ```bash
   python upload_datasets_to_s3.py
   ```

4. **Test on Trainium**:
   ```python
   from dataset_loader import load_dataset
   data = load_dataset('my_dataset')
   ```

## References

- AWS S3 Documentation: https://docs.aws.amazon.com/s3/
- PyTorch Datasets: https://pytorch.org/vision/stable/datasets.html
- HuggingFace Datasets: https://huggingface.co/docs/datasets/
- Trainium Setup: `trainium_executor/README.md`

---

**Questions?** Check the main README or contact the team.

