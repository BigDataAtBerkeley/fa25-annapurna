#!/usr/bin/env python3
"""
Upload Popular ML Datasets to S3

This script downloads commonly-used ML datasets and uploads them to the
'datasets-for-all-papers' S3 bucket for use by Trainium instances.

Datasets included:
- CIFAR-10/100 (Computer Vision)
- MNIST/FashionMNIST (Simple CV)
- IMDB Reviews (NLP)
- WikiText-2 (Language Modeling)
- ImageNet subset (optional, large)
- Synthetic test datasets
"""

import os
import sys
import boto3
import torch
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
import tarfile
import json
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S3_BUCKET = 'datasets-for-all-papers'
LOCAL_CACHE = './dataset_cache'
os.makedirs(LOCAL_CACHE, exist_ok=True)

s3_client = boto3.client('s3')

def upload_directory_to_s3(local_path: str, s3_prefix: str):
    """Upload entire directory to S3 with progress bar"""
    files_to_upload = []
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            s3_key = f"{s3_prefix}/{relative_path}"
            files_to_upload.append((local_file, s3_key))
    
    logger.info(f"Uploading {len(files_to_upload)} files from {local_path} to s3://{S3_BUCKET}/{s3_prefix}")
    
    for local_file, s3_key in tqdm(files_to_upload, desc=f"Uploading {s3_prefix}"):
        s3_client.upload_file(local_file, S3_BUCKET, s3_key)
    
    logger.info(f"✓ Uploaded {s3_prefix} to S3")

def upload_file_to_s3(local_file: str, s3_key: str):
    """Upload single file to S3"""
    logger.info(f"Uploading {local_file} to s3://{S3_BUCKET}/{s3_key}")
    s3_client.upload_file(local_file, S3_BUCKET, s3_key)
    logger.info(f"✓ Uploaded {s3_key}")

def create_dataset_metadata(name: str, dataset_type: str, size_mb: float, 
                           num_samples: int, description: str, s3_prefix: str,
                           usage_example: str):
    """Create metadata file for dataset"""
    metadata = {
        "name": name,
        "type": dataset_type,
        "size_mb": size_mb,
        "num_samples": num_samples,
        "description": description,
        "s3_bucket": S3_BUCKET,
        "s3_prefix": s3_prefix,
        "usage_example": usage_example,
        "uploaded_at": "2025-10-30"
    }
    
    metadata_file = os.path.join(LOCAL_CACHE, f"{name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, indent=2, fp=f)
    
    upload_file_to_s3(metadata_file, f"{s3_prefix}/metadata.json")
    return metadata

def download_cifar10():
    """Download and upload CIFAR-10"""
    logger.info("=" * 50)
    logger.info("Downloading CIFAR-10...")
    
    dataset_dir = os.path.join(LOCAL_CACHE, 'cifar10')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download train and test
    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, 
        train=True, 
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, 
        train=False, 
        download=True
    )
    
    # Extract raw data as tensors (avoid saving dataset objects to prevent torchvision import issues)
    import torchvision.transforms as transforms
    to_tensor = transforms.ToTensor()
    
    train_images = torch.stack([to_tensor(train_dataset[i][0]) for i in range(len(train_dataset))])
    train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    test_images = torch.stack([to_tensor(test_dataset[i][0]) for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    
    # Save as raw tensors (no torchvision dependencies)
    torch_file = os.path.join(dataset_dir, 'cifar10_pytorch.pt')
    torch.save({
        'train_data': train_images,
        'train_labels': train_labels,
        'test_data': test_images,
        'test_labels': test_labels,
        'num_classes': 10,
        'classes': train_dataset.classes
    }, torch_file)
    
    # Upload to S3
    upload_directory_to_s3(dataset_dir, 'cifar10')
    
    # Get size
    size_mb = sum(f.stat().st_size for f in Path(dataset_dir).rglob('*') if f.is_file()) / (1024 * 1024)
    
    create_dataset_metadata(
        name="cifar10",
        dataset_type="image_classification",
        size_mb=round(size_mb, 2),
        num_samples=60000,
        description="CIFAR-10: 60K 32x32 color images in 10 classes (50K train, 10K test)",
        s3_prefix="cifar10",
        usage_example="""
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10(root='./data/cifar10', train=True, transform=transform)
test_dataset = CIFAR10(root='./data/cifar10', train=False, transform=transform)
"""
    )

def download_cifar100():
    """Download and upload CIFAR-100"""
    logger.info("=" * 50)
    logger.info("Downloading CIFAR-100...")
    
    dataset_dir = os.path.join(LOCAL_CACHE, 'cifar100')
    os.makedirs(dataset_dir, exist_ok=True)
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=dataset_dir, 
        train=True, 
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=dataset_dir, 
        train=False, 
        download=True
    )
    
    # Extract raw data as tensors
    import torchvision.transforms as transforms
    to_tensor = transforms.ToTensor()
    
    train_images = torch.stack([to_tensor(train_dataset[i][0]) for i in range(len(train_dataset))])
    train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    test_images = torch.stack([to_tensor(test_dataset[i][0]) for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    
    torch_file = os.path.join(dataset_dir, 'cifar100_pytorch.pt')
    torch.save({
        'train_data': train_images,
        'train_labels': train_labels,
        'test_data': test_images,
        'test_labels': test_labels,
        'num_classes': 100
    }, torch_file)
    
    upload_directory_to_s3(dataset_dir, 'cifar100')
    
    size_mb = sum(f.stat().st_size for f in Path(dataset_dir).rglob('*') if f.is_file()) / (1024 * 1024)
    
    create_dataset_metadata(
        name="cifar100",
        dataset_type="image_classification",
        size_mb=round(size_mb, 2),
        num_samples=60000,
        description="CIFAR-100: 60K 32x32 color images in 100 classes",
        s3_prefix="cifar100",
        usage_example="Same as CIFAR-10, use CIFAR100 class"
    )

def download_mnist():
    """Download and upload MNIST"""
    logger.info("=" * 50)
    logger.info("Downloading MNIST...")
    
    dataset_dir = os.path.join(LOCAL_CACHE, 'mnist')
    os.makedirs(dataset_dir, exist_ok=True)
    
    train_dataset = torchvision.datasets.MNIST(
        root=dataset_dir, 
        train=True, 
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=dataset_dir, 
        train=False, 
        download=True
    )
    
    # Extract raw data as tensors
    import torchvision.transforms as transforms
    to_tensor = transforms.ToTensor()
    
    train_images = torch.stack([to_tensor(train_dataset[i][0]) for i in range(len(train_dataset))])
    train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    test_images = torch.stack([to_tensor(test_dataset[i][0]) for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    
    # Save as raw tensors
    torch_file = os.path.join(dataset_dir, 'mnist_pytorch.pt')
    torch.save({
        'train_data': train_images,
        'train_labels': train_labels,
        'test_data': test_images,
        'test_labels': test_labels,
        'num_classes': 10
    }, torch_file)
    
    upload_directory_to_s3(dataset_dir, 'mnist')
    
    size_mb = sum(f.stat().st_size for f in Path(dataset_dir).rglob('*') if f.is_file()) / (1024 * 1024)
    
    create_dataset_metadata(
        name="mnist",
        dataset_type="image_classification",
        size_mb=round(size_mb, 2),
        num_samples=70000,
        description="MNIST: 70K grayscale handwritten digit images (60K train, 10K test)",
        s3_prefix="mnist",
        usage_example="from torchvision.datasets import MNIST"
    )

def download_fashion_mnist():
    """Download and upload Fashion-MNIST"""
    logger.info("=" * 50)
    logger.info("Downloading Fashion-MNIST...")
    
    dataset_dir = os.path.join(LOCAL_CACHE, 'fashion_mnist')
    os.makedirs(dataset_dir, exist_ok=True)
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root=dataset_dir, 
        train=True, 
        download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=dataset_dir, 
        train=False, 
        download=True
    )
    
    # Extract raw data as tensors
    import torchvision.transforms as transforms
    to_tensor = transforms.ToTensor()
    
    train_images = torch.stack([to_tensor(train_dataset[i][0]) for i in range(len(train_dataset))])
    train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    test_images = torch.stack([to_tensor(test_dataset[i][0]) for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    
    # Save as raw tensors
    torch_file = os.path.join(dataset_dir, 'fashion_mnist_pytorch.pt')
    torch.save({
        'train_data': train_images,
        'train_labels': train_labels,
        'test_data': test_images,
        'test_labels': test_labels,
        'num_classes': 10
    }, torch_file)
    
    upload_directory_to_s3(dataset_dir, 'fashion_mnist')
    
    size_mb = sum(f.stat().st_size for f in Path(dataset_dir).rglob('*') if f.is_file()) / (1024 * 1024)
    
    create_dataset_metadata(
        name="fashion_mnist",
        dataset_type="image_classification",
        size_mb=round(size_mb, 2),
        num_samples=70000,
        description="Fashion-MNIST: 70K grayscale clothing images in 10 categories",
        s3_prefix="fashion_mnist",
        usage_example="from torchvision.datasets import FashionMNIST"
    )

def download_imdb():
    """Download and upload IMDB Reviews dataset"""
    logger.info("=" * 50)
    logger.info("Downloading IMDB Reviews...")
    
    dataset_dir = os.path.join(LOCAL_CACHE, 'imdb')
    os.makedirs(dataset_dir, exist_ok=True)
    
    try:
        dataset = load_dataset("imdb")
        dataset.save_to_disk(dataset_dir)
        
        upload_directory_to_s3(dataset_dir, 'imdb')
        
        size_mb = sum(f.stat().st_size for f in Path(dataset_dir).rglob('*') if f.is_file()) / (1024 * 1024)
        
        create_dataset_metadata(
            name="imdb",
            dataset_type="text_classification",
            size_mb=round(size_mb, 2),
            num_samples=50000,
            description="IMDB Reviews: 50K movie reviews for sentiment classification",
            s3_prefix="imdb",
            usage_example="""
from datasets import load_from_disk
dataset = load_from_disk('./data/imdb')
train = dataset['train']
test = dataset['test']
"""
        )
    except Exception as e:
        logger.error(f"Failed to download IMDB: {e}")

def download_wikitext():
    """Download and upload WikiText-2"""
    logger.info("=" * 50)
    logger.info("Downloading WikiText-2...")
    
    dataset_dir = os.path.join(LOCAL_CACHE, 'wikitext2')
    os.makedirs(dataset_dir, exist_ok=True)
    
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        dataset.save_to_disk(dataset_dir)
        
        upload_directory_to_s3(dataset_dir, 'wikitext2')
        
        size_mb = sum(f.stat().st_size for f in Path(dataset_dir).rglob('*') if f.is_file()) / (1024 * 1024)
        
        create_dataset_metadata(
            name="wikitext2",
            dataset_type="language_modeling",
            size_mb=round(size_mb, 2),
            num_samples=36718,
            description="WikiText-2: Language modeling dataset from Wikipedia",
            s3_prefix="wikitext2",
            usage_example="from datasets import load_from_disk; dataset = load_from_disk('./data/wikitext2')"
        )
    except Exception as e:
        logger.error(f"Failed to download WikiText-2: {e}")

def create_synthetic_datasets():
    """Create synthetic datasets for testing"""
    logger.info("=" * 50)
    logger.info("Creating synthetic test datasets...")
    
    dataset_dir = os.path.join(LOCAL_CACHE, 'synthetic')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Small synthetic dataset for quick testing
    small_data = {
        'images': torch.randn(1000, 3, 224, 224),
        'labels': torch.randint(0, 10, (1000,)),
        'description': 'Small synthetic dataset: 1K samples, 224x224 RGB images, 10 classes'
    }
    torch.save(small_data, os.path.join(dataset_dir, 'synthetic_small.pt'))
    
    # Medium synthetic dataset
    medium_data = {
        'images': torch.randn(10000, 3, 224, 224),
        'labels': torch.randint(0, 100, (10000,)),
        'description': 'Medium synthetic dataset: 10K samples, 224x224 RGB images, 100 classes'
    }
    torch.save(medium_data, os.path.join(dataset_dir, 'synthetic_medium.pt'))
    
    # Tabular synthetic dataset
    tabular_data = {
        'features': torch.randn(5000, 50),
        'labels': torch.randint(0, 2, (5000,)),
        'description': 'Tabular synthetic dataset: 5K samples, 50 features, binary classification'
    }
    torch.save(tabular_data, os.path.join(dataset_dir, 'synthetic_tabular.pt'))
    
    upload_directory_to_s3(dataset_dir, 'synthetic')
    
    size_mb = sum(f.stat().st_size for f in Path(dataset_dir).rglob('*') if f.is_file()) / (1024 * 1024)
    
    create_dataset_metadata(
        name="synthetic",
        dataset_type="synthetic",
        size_mb=round(size_mb, 2),
        num_samples=16000,
        description="Synthetic test datasets for quick testing (small, medium, tabular)",
        s3_prefix="synthetic",
        usage_example="data = torch.load('./data/synthetic/synthetic_small.pt')"
    )

def create_master_index():
    """Create master index of all datasets"""
    logger.info("=" * 50)
    logger.info("Creating master dataset index...")
    
    # List all metadata files from S3
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET)
    
    datasets_info = []
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('metadata.json'):
            # Download metadata
            metadata_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=obj['Key'])
            metadata = json.loads(metadata_obj['Body'].read().decode('utf-8'))
            datasets_info.append(metadata)
    
    index = {
        "total_datasets": len(datasets_info),
        "total_size_mb": sum(d['size_mb'] for d in datasets_info),
        "datasets": datasets_info,
        "bucket": S3_BUCKET,
        "last_updated": "2025-10-30"
    }
    
    index_file = os.path.join(LOCAL_CACHE, 'dataset_index.json')
    with open(index_file, 'w') as f:
        json.dump(index, indent=2, fp=f)
    
    upload_file_to_s3(index_file, 'dataset_index.json')
    
    logger.info(f"✓ Created master index with {len(datasets_info)} datasets")
    logger.info(f"  Total size: {index['total_size_mb']:.2f} MB")
    
    return index

def main():
    """Main function to download and upload all datasets"""
    logger.info("Starting dataset upload to S3...")
    logger.info(f"Target bucket: s3://{S3_BUCKET}")
    logger.info(f"Local cache: {LOCAL_CACHE}")
    
    try:
        # Verify S3 bucket exists
        s3_client.head_bucket(Bucket=S3_BUCKET)
        logger.info(f"✓ S3 bucket '{S3_BUCKET}' is accessible")
    except Exception as e:
        logger.error(f"Cannot access S3 bucket '{S3_BUCKET}': {e}")
        return
    
    # Download and upload each dataset
    datasets = {
        'CIFAR-10': download_cifar10,
        'CIFAR-100': download_cifar100,
        'MNIST': download_mnist,
        'Fashion-MNIST': download_fashion_mnist,
        'IMDB': download_imdb,
        'WikiText-2': download_wikitext,
        'Synthetic': create_synthetic_datasets
    }
    
    for name, func in datasets.items():
        try:
            func()
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")
            continue
    
    # Create master index
    index = create_master_index()
    
    logger.info("=" * 50)
    logger.info("✓ All datasets uploaded successfully!")
    logger.info(f"Total datasets: {index['total_datasets']}")
    logger.info(f"Total size: {index['total_size_mb']:.2f} MB")
    logger.info(f"Bucket: s3://{S3_BUCKET}")
    logger.info("\nDatasets available:")
    for ds in index['datasets']:
        logger.info(f"  - {ds['name']}: {ds['size_mb']:.2f} MB ({ds['num_samples']} samples)")

if __name__ == '__main__':
    main()

