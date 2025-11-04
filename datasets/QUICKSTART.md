# Dataset Pipeline Quickstart

## 🚀 Quick Setup (5 minutes)

### 1. Upload Datasets to S3
```bash
cd /Users/danchizik/Desktop/annapurna/datasets
source ../aws_env/bin/activate
pip install boto3 torch torchvision datasets tqdm

# Upload all datasets (takes ~15-30 min)
python upload_datasets_to_s3.py
```

### 2. Test the Pipeline
```bash
# Run test suite
python test_dataset_pipeline.py
```

### 3. Deploy to Trainium
```bash
cd ../deployment
./deploy_trainium.sh /path/to/your-key.pem
```

---

## 💻 Using Datasets in Generated Code

### Basic Usage
```python
from dataset_loader import load_dataset

# Load CIFAR-10 for computer vision
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# Load IMDB for NLP
train_data, test_data = load_dataset('imdb')

# Load synthetic data for quick testing
data = load_dataset('synthetic', variant='small')
```

### List Available Datasets
```python
from dataset_loader import list_available_datasets

status = list_available_datasets()
print(f"Ready to use: {status['available']}")
# Output: ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'imdb', 'wikitext2', 'synthetic']
```

### Full Training Example
```python
from dataset_loader import load_dataset
import torch.nn as nn
import torch.optim as optim

# 1. Load data
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# 2. Define model
model = MyCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Train
for epoch in range(10):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"METRICS: {{'epoch': {epoch}, 'loss': {loss.item()}}}")
```

---

## 🔧 Adding Custom Datasets

### Method 1: Register a Loader Dynamically
```python
from dataset_loader import register_custom_loader, load_dataset
import torch
import os

def load_my_custom_dataset(dataset_dir, batch_size=64, **kwargs):
    """Custom loader for your dataset"""
    # Load from downloaded S3 files
    data_file = os.path.join(dataset_dir, 'data.pt')
    data = torch.load(data_file)
    
    # Return train/test loaders
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    test_dataset = TensorDataset(data['X_test'], data['y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Register it
register_custom_loader('my_dataset', load_my_custom_dataset)

# Use it
train, test = load_dataset('my_dataset', batch_size=128)
```

### Method 2: Upload New Dataset to S3

**Step 1:** Add to `upload_datasets_to_s3.py`:
```python
def download_imagenet_subset():
    """Download and upload ImageNet subset"""
    logger.info("Downloading ImageNet subset...")
    
    dataset_dir = os.path.join(LOCAL_CACHE, 'imagenet_subset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download your dataset
    # ... your download logic ...
    
    # Upload to S3
    upload_directory_to_s3(dataset_dir, 'imagenet_subset')
    
    create_dataset_metadata(
        name="imagenet_subset",
        dataset_type="image_classification",
        size_mb=1024.0,
        num_samples=100000,
        description="ImageNet subset: 100K images",
        s3_prefix="imagenet_subset",
        usage_example="from dataset_loader import load_dataset; data = load_dataset('imagenet_subset')"
    )
```

**Step 2:** Add loader to `dataset_loader.py`:
```python
# In DatasetManager._build_loader_registry():
def _build_loader_registry(self):
    return {
        'cifar10': self._load_cifar10,
        # ... existing loaders ...
        'imagenet_subset': self._load_imagenet_subset  # Add this
    }

# Add the loader method:
def _load_imagenet_subset(self, dataset_dir: str, batch_size: int = 128, **kwargs):
    """Load ImageNet subset"""
    # Your loading logic
    pass
```

**Step 3:** Upload and test:
```bash
python upload_datasets_to_s3.py
python test_dataset_pipeline.py
```

---

## 🎯 Dataset Recommendations by Paper Type

| Paper Type | Recommended Dataset | Reason |
|------------|-------------------|--------|
| **CNN / Vision** | `cifar10` | Standard benchmark, 32x32 images |
| **ResNet / Deep Vision** | `cifar100` | More classes, harder task |
| **Simple Neural Nets** | `mnist` | Fast, easy to debug |
| **Transformers / Attention** | `imdb` or `wikitext2` | Text data for NLP |
| **Language Models** | `wikitext2` | Pre-tokenized, clean |
| **Quick Debugging** | `synthetic` (small) | Instant load, no download |
| **Performance Testing** | `synthetic` (medium) | Larger, controlled size |

---

## 📊 S3 Bucket Structure

```
s3://datasets-for-all-papers/
├── dataset_index.json              # Master index
├── cifar10/
│   ├── metadata.json              # Dataset info
│   ├── cifar-10-batches-py/       # Raw CIFAR files
│   └── cifar10_pytorch.pt         # Preprocessed tensor
├── mnist/
│   ├── metadata.json
│   └── MNIST/raw/...
├── imdb/
│   ├── metadata.json
│   └── dataset files...
└── synthetic/
    ├── metadata.json
    ├── synthetic_small.pt
    ├── synthetic_medium.pt
    └── synthetic_tabular.pt
```

---

## 🔍 API Reference

### `load_dataset(name, **kwargs)`
Load a dataset by name. Returns train/test loaders or dataset objects.

**Args:**
- `name` (str): Dataset name ('cifar10', 'mnist', etc.)
- `batch_size` (int): Batch size for DataLoader (default: 128)
- `**kwargs`: Additional dataset-specific args

**Returns:**
- `(train_loader, test_loader)` for vision datasets
- `(train_data, test_data)` for NLP datasets
- `dict` for synthetic datasets

**Example:**
```python
train_loader, test_loader = load_dataset('cifar10', batch_size=64)
```

### `list_available_datasets()`
Returns dict with dataset availability info.

**Returns:**
```python
{
    'available': ['cifar10', 'mnist', ...],  # Ready to use
    'in_s3': ['cifar10', 'mnist', ...],      # In S3 bucket
    'registered': ['cifar10', 'mnist', ...]  # Have loaders
}
```

### `register_custom_loader(name, loader_func)`
Register a custom dataset loader.

**Args:**
- `name` (str): Dataset identifier
- `loader_func` (callable): Function(dataset_dir, **kwargs) -> dataset

**Example:**
```python
def my_loader(dataset_dir, **kwargs):
    return load_from_disk(dataset_dir)

register_custom_loader('my_dataset', my_loader)
```

### `download_dataset(name, force=False)`
Download dataset from S3 to local cache.

**Args:**
- `name` (str): Dataset name
- `force` (bool): Force re-download even if cached

**Returns:**
- `str`: Local path to dataset directory

---

## ⚡ Performance Tips

1. **Cache datasets on Trainium**: Datasets persist in `/tmp/datasets` across executions
2. **Use smaller batch sizes** for faster iteration during debugging
3. **Start with synthetic data** to test code before using real datasets
4. **Check S3 region**: Trainium should be in same region as bucket (us-east-1)

---

## 🐛 Troubleshooting

### "Unknown dataset" error
```python
# Check what's available
from dataset_loader import list_available_datasets
print(list_available_datasets())

# Register if needed
from dataset_loader import register_custom_loader
register_custom_loader('my_dataset', my_loader_func)
```

### Dataset download fails
```bash
# Check S3 access
aws s3 ls s3://datasets-for-all-papers/

# Check IAM permissions on Trainium
aws sts get-caller-identity
```

### Slow downloads
```bash
# Pre-warm cache on Trainium
ssh ubuntu@<TRAINIUM_IP>
python3 -c "from dataset_loader import download_dataset; download_dataset('cifar10')"
```

---

## 📈 Next Steps

1. ✅ Upload datasets: `python upload_datasets_to_s3.py`
2. ✅ Test locally: `python test_dataset_pipeline.py`
3. ✅ Deploy to Trainium: `./deploy_trainium.sh`
4. 🔄 Update code generation to use `load_dataset()`
5. 🧪 Test generated code on Trainium
6. 📊 Monitor results in OpenSearch

**Questions?** See the full docs in `datasets/README.md`

