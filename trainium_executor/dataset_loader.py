#!/usr/bin/env python3
"""
Dataset Loader for Trainium Executor

This module provides utilities to download and load datasets from S3
for use by generated PyTorch code on Trainium instances.
"""

import os
import boto3
import torch
import torch.nn.functional as F
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

S3_BUCKET = 'datasets-for-all-papers'
LOCAL_DATA_DIR = os.getenv('DATASET_CACHE_DIR', '/tmp/datasets')
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

s3_client = boto3.client('s3')


class SimpleImageDataset(Dataset):
    """Simple PyTorch Dataset for image data without torchvision dependencies"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to float and normalize if needed
        if image.dtype != torch.float32:
            image = image.float() / 255.0
        
        # Apply simple transforms using torch operations
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DatasetManager:
    """Manages dataset downloads and caching from S3"""
    
    def __init__(self, cache_dir: str = LOCAL_DATA_DIR):
        self.cache_dir = cache_dir
        self.s3_bucket = S3_BUCKET
        os.makedirs(self.cache_dir, exist_ok=True)
        self._index = None
        self._loader_registry = self._build_loader_registry()
    
    def _build_loader_registry(self):
        """Build a registry of dataset loaders dynamically"""
        return {
            'cifar10': self._load_cifar10,
            'cifar100': self._load_cifar100,
            'mnist': self._load_mnist,
            'fashion_mnist': self._load_fashion_mnist,
            'imdb': self._load_imdb,  # Uses HuggingFace datasets.load_from_disk
            'wikitext2': self._load_wikitext  # Uses HuggingFace datasets.load_from_disk
        }
    
    def register_dataset_loader(self, name: str, loader_func):
        """
        Register a custom dataset loader.
        
        Args:
            name: Dataset name
            loader_func: Function that takes (dataset_dir, **kwargs) and returns dataset
            
        Example:
            manager = DatasetManager()
            manager.register_dataset_loader('custom_dataset', my_loader_func)
        """
        self._loader_registry[name] = loader_func
        logger.info(f"Registered custom loader for dataset: {name}")
    
    def get_dataset_index(self) -> Dict[str, Any]:
        """Download and cache the master dataset index"""
        if self._index is None:
            index_file = os.path.join(self.cache_dir, 'dataset_index.json')
            
            if not os.path.exists(index_file):
                logger.info("Downloading dataset index from S3...")
                try:
                    s3_client.download_file(
                        self.s3_bucket,
                        'dataset_index.json',
                        index_file
                    )
                except Exception as e:
                    logger.warning(f"Failed to download dataset index: {e}")
                    return {"datasets": []}
            
            with open(index_file, 'r') as f:
                self._index = json.load(f)
        
        return self._index
    
    def list_datasets(self) -> list:
        """List all available datasets"""
        index = self.get_dataset_index()
        return [ds['name'] for ds in index.get('datasets', [])]
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> str:
        """
        Download dataset from S3 to local cache.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'cifar10', 'imdb')
            force: Force re-download even if cached
            
        Returns:
            Local path to the dataset directory
        """
        dataset_dir = os.path.join(self.cache_dir, dataset_name)
        
        if os.path.exists(dataset_dir) and not force:
            logger.info(f"Dataset '{dataset_name}' already cached at {dataset_dir}")
            return dataset_dir
        
        os.makedirs(dataset_dir, exist_ok=True)
        
        logger.info(f"Downloading dataset '{dataset_name}' from S3...")
        
        try:
            # List all files in the dataset prefix
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=f"{dataset_name}/")
            
            file_count = 0
            for page in pages:
                for obj in page.get('Contents', []):
                    s3_key = obj['Key']
                    
                    # Skip the prefix itself
                    if s3_key == f"{dataset_name}/":
                        continue
                    
                    # Determine local file path
                    relative_path = s3_key[len(dataset_name)+1:]  # Remove prefix
                    local_file = os.path.join(dataset_dir, relative_path)
                    
                    # Create parent directories
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    
                    # Download file
                    s3_client.download_file(self.s3_bucket, s3_key, local_file)
                    file_count += 1
            
            logger.info(f"✓ Downloaded {file_count} files for '{dataset_name}' to {dataset_dir}")
            return dataset_dir
            
        except Exception as e:
            logger.error(f"Failed to download dataset '{dataset_name}': {e}")
            raise
    
    def load_dataset(self, dataset_name: str, **kwargs):
        """
        Load dataset into memory using registered loaders.
        
        Args:
            dataset_name: Name of the dataset
            **kwargs: Additional arguments for dataset loading
            
        Returns:
            Dataset object or tuple of (train, test)
            
        Raises:
            ValueError: If dataset name is not registered
        """
        # Check if loader is registered
        if dataset_name not in self._loader_registry:
            available = ', '.join(self._loader_registry.keys())
            raise ValueError(
                f"Unknown dataset: '{dataset_name}'. "
                f"Available datasets: {available}. "
                f"Use register_dataset_loader() to add custom loaders."
            )
        
        # Download dataset from S3
        dataset_dir = self.download_dataset(dataset_name)
        
        # Call the registered loader function
        loader_func = self._loader_registry[dataset_name]
        return loader_func(dataset_dir, **kwargs)
    
    def _load_cifar10(self, dataset_dir: str, batch_size: int = 128, **kwargs):
        """Load CIFAR-10 dataset from pre-processed .pt file"""
        pt_file = os.path.join(dataset_dir, 'cifar10_pytorch.pt')
        
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"CIFAR-10 data file not found: {pt_file}. Please download from S3 first.")
        
        # Load pre-processed data
        data = torch.load(pt_file, map_location='cpu')
        
        # Extract data from torchvision dataset objects if needed
        # The .pt file contains the dataset objects, so we need to extract the actual data
        try:
            # Try to get data directly if it's stored as tensors
            if isinstance(data, dict) and 'train_data' in data:
                train_images = data['train_data']
                train_labels = data['train_labels']
                test_images = data['test_data']
                test_labels = data['test_labels']
            else:
                # Extract from dataset objects
                train_dataset_obj = data.get('train')
                test_dataset_obj = data.get('test')
                
                # Get all data from dataset objects
                train_images = torch.stack([torch.tensor(train_dataset_obj[i][0]) for i in range(len(train_dataset_obj))])
                train_labels = torch.tensor([train_dataset_obj[i][1] for i in range(len(train_dataset_obj))])
                test_images = torch.stack([torch.tensor(test_dataset_obj[i][0]) for i in range(len(test_dataset_obj))])
                test_labels = torch.tensor([test_dataset_obj[i][1] for i in range(len(test_dataset_obj))])
        except Exception as e:
            logger.error(f"Error extracting CIFAR-10 data: {e}")
            raise
        
        # Normalize: (x / 255.0 - 0.5) / 0.5 = (x - 127.5) / 127.5
        def normalize_transform(img):
            if img.max() > 1.0:
                img = img.float() / 255.0
            return (img - 0.5) / 0.5
        
        # Create simple datasets
        train_dataset = SimpleImageDataset(train_images, train_labels, transform=normalize_transform)
        test_dataset = SimpleImageDataset(test_images, test_labels, transform=normalize_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    
    def _load_cifar100(self, dataset_dir: str, batch_size: int = 128, **kwargs):
        """Load CIFAR-100 dataset from pre-processed .pt file"""
        pt_file = os.path.join(dataset_dir, 'cifar100_pytorch.pt')
        
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"CIFAR-100 data file not found: {pt_file}. Please download from S3 first.")
        
        data = torch.load(pt_file, map_location='cpu')
        
        try:
            if isinstance(data, dict) and 'train_data' in data:
                train_images = data['train_data']
                train_labels = data['train_labels']
                test_images = data['test_data']
                test_labels = data['test_labels']
            else:
                train_dataset_obj = data.get('train')
                test_dataset_obj = data.get('test')
                train_images = torch.stack([torch.tensor(train_dataset_obj[i][0]) for i in range(len(train_dataset_obj))])
                train_labels = torch.tensor([train_dataset_obj[i][1] for i in range(len(train_dataset_obj))])
                test_images = torch.stack([torch.tensor(test_dataset_obj[i][0]) for i in range(len(test_dataset_obj))])
                test_labels = torch.tensor([test_dataset_obj[i][1] for i in range(len(test_dataset_obj))])
        except Exception as e:
            logger.error(f"Error extracting CIFAR-100 data: {e}")
            raise
        
        def normalize_transform(img):
            if img.max() > 1.0:
                img = img.float() / 255.0
            return (img - 0.5) / 0.5
        
        train_dataset = SimpleImageDataset(train_images, train_labels, transform=normalize_transform)
        test_dataset = SimpleImageDataset(test_images, test_labels, transform=normalize_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    
    def _load_mnist(self, dataset_dir: str, batch_size: int = 128, **kwargs):
        """Load MNIST dataset from pre-processed .pt file"""
        pt_file = os.path.join(dataset_dir, 'mnist_pytorch.pt')
        
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"MNIST data file not found: {pt_file}. Please download from S3 first.")
        
        data = torch.load(pt_file, map_location='cpu')
        
        try:
            if isinstance(data, dict) and 'train_data' in data:
                train_images = data['train_data']
                train_labels = data['train_labels']
                test_images = data['test_data']
                test_labels = data['test_labels']
            else:
                train_dataset_obj = data.get('train')
                test_dataset_obj = data.get('test')
                train_images = torch.stack([torch.tensor(train_dataset_obj[i][0]) for i in range(len(train_dataset_obj))])
                train_labels = torch.tensor([train_dataset_obj[i][1] for i in range(len(train_dataset_obj))])
                test_images = torch.stack([torch.tensor(test_dataset_obj[i][0]) for i in range(len(test_dataset_obj))])
                test_labels = torch.tensor([test_dataset_obj[i][1] for i in range(len(test_dataset_obj))])
        except Exception as e:
            logger.error(f"Error extracting MNIST data: {e}")
            raise
        
        # MNIST normalization: mean=0.1307, std=0.3081
        def normalize_transform(img):
            if img.max() > 1.0:
                img = img.float() / 255.0
            return (img - 0.1307) / 0.3081
        
        train_dataset = SimpleImageDataset(train_images, train_labels, transform=normalize_transform)
        test_dataset = SimpleImageDataset(test_images, test_labels, transform=normalize_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    
    def _load_fashion_mnist(self, dataset_dir: str, batch_size: int = 128, **kwargs):
        """Load Fashion-MNIST dataset from pre-processed .pt file"""
        pt_file = os.path.join(dataset_dir, 'fashion_mnist_pytorch.pt')
        
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"Fashion-MNIST data file not found: {pt_file}. Please download from S3 first.")
        
        data = torch.load(pt_file, map_location='cpu')
        
        try:
            if isinstance(data, dict) and 'train_data' in data:
                train_images = data['train_data']
                train_labels = data['train_labels']
                test_images = data['test_data']
                test_labels = data['test_labels']
            else:
                train_dataset_obj = data.get('train')
                test_dataset_obj = data.get('test')
                train_images = torch.stack([torch.tensor(train_dataset_obj[i][0]) for i in range(len(train_dataset_obj))])
                train_labels = torch.tensor([train_dataset_obj[i][1] for i in range(len(train_dataset_obj))])
                test_images = torch.stack([torch.tensor(test_dataset_obj[i][0]) for i in range(len(test_dataset_obj))])
                test_labels = torch.tensor([test_dataset_obj[i][1] for i in range(len(test_dataset_obj))])
        except Exception as e:
            logger.error(f"Error extracting Fashion-MNIST data: {e}")
            raise
        
        def normalize_transform(img):
            if img.max() > 1.0:
                img = img.float() / 255.0
            return (img - 0.5) / 0.5
        
        train_dataset = SimpleImageDataset(train_images, train_labels, transform=normalize_transform)
        test_dataset = SimpleImageDataset(test_images, test_labels, transform=normalize_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    
    def _load_imdb(self, dataset_dir: str, batch_size: int = 128, **kwargs):
        """Load IMDB dataset from HuggingFace Arrow format (as per AWS Trainium best practices)"""
        # Check if _lzma is available (required by datasets package)
        try:
            import _lzma
        except ImportError:
            raise ImportError(
                "The 'datasets' package requires the '_lzma' module, which is not available "
                "in the Neuron venv's Python. IMDB dataset cannot be loaded on this system. "
                "Please use a different dataset (e.g., mnist, cifar10, cifar100, fashion_mnist)."
            )
        
        try:
            from datasets import load_from_disk
        except Exception as e:
            raise ImportError(
                f"HuggingFace 'datasets' package is required for IMDB but failed to import: {e}. "
                "Please use a different dataset (e.g., mnist, cifar10, cifar100, fashion_mnist)."
            )
        
        # Load from Arrow format (as stored in S3)
        dataset = load_from_disk(dataset_dir)
        
        # Convert to PyTorch DataLoader
        # IMDB has 'text' and 'label' columns
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        
        # Create simple dataset wrapper for DataLoader
        class IMDBDataset(Dataset):
            def __init__(self, hf_dataset):
                self.hf_dataset = hf_dataset
            
            def __len__(self):
                return len(self.hf_dataset)
            
            def __getitem__(self, idx):
                item = self.hf_dataset[idx]
                # Return text and label - tokenization should be done in the model code
                return item['text'], item['label']
        
        train_pytorch_dataset = IMDBDataset(train_dataset)
        test_pytorch_dataset = IMDBDataset(test_dataset)
        
        train_loader = DataLoader(train_pytorch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_pytorch_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    
    def _load_wikitext(self, dataset_dir: str, batch_size: int = 128, **kwargs):
        """Load WikiText-2 dataset from HuggingFace Arrow format (as per AWS Trainium best practices)"""
        # Check if _lzma is available (required by datasets package)
        try:
            import _lzma
        except ImportError:
            raise ImportError(
                "The 'datasets' package requires the '_lzma' module, which is not available "
                "in the Neuron venv's Python. WikiText-2 dataset cannot be loaded on this system. "
                "Please use a different dataset (e.g., mnist, cifar10, cifar100, fashion_mnist)."
            )
        
        try:
            from datasets import load_from_disk
        except Exception as e:
            raise ImportError(
                f"HuggingFace 'datasets' package is required for WikiText-2 but failed to import: {e}. "
                "Please use a different dataset (e.g., mnist, cifar10, cifar100, fashion_mnist)."
            )
        
        # Load from Arrow format (as stored in S3)
        dataset = load_from_disk(dataset_dir)
        
        # WikiText-2 has 'text' column
        train_dataset = dataset.get('train', dataset.get('validation'))
        
        # Create simple dataset wrapper
        class WikiTextDataset(Dataset):
            def __init__(self, hf_dataset):
                self.hf_dataset = hf_dataset
                # Filter out empty texts
                self.valid_indices = [i for i, item in enumerate(hf_dataset) if len(item.get('text', '').strip()) > 0]
            
            def __len__(self):
                return len(self.valid_indices)
            
            def __getitem__(self, idx):
                actual_idx = self.valid_indices[idx]
                item = self.hf_dataset[actual_idx]
                return item['text']
        
        train_pytorch_dataset = WikiTextDataset(train_dataset)
        val_pytorch_dataset = WikiTextDataset(dataset.get('validation', train_dataset[:1000]))
        
        train_loader = DataLoader(train_pytorch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_pytorch_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader


# Global dataset manager instance
_dataset_manager = None

def get_dataset_manager() -> DatasetManager:
    """Get or create the global dataset manager"""
    global _dataset_manager
    if _dataset_manager is None:
        _dataset_manager = DatasetManager()
    return _dataset_manager


# Convenience functions for generated code
def load_dataset(name: str, **kwargs):
    """
    Convenience function to load a dataset.
    
    Usage in generated code:
        from dataset_loader import load_dataset
        train_loader, test_loader = load_dataset('cifar10', batch_size=128)
    """
    manager = get_dataset_manager()
    return manager.load_dataset(name, **kwargs)


def list_available_datasets():
    """List all available datasets"""
    manager = get_dataset_manager()
    
    # Get datasets from S3 index
    s3_datasets = manager.list_datasets()
    
    # Get datasets with registered loaders
    registered = list(manager._loader_registry.keys())
    
    # Return datasets that are both in S3 and have loaders
    available = [ds for ds in s3_datasets if ds in registered]
    
    return {
        'available': available,  # Ready to use
        'in_s3': s3_datasets,    # In S3 but may not have loader
        'registered': registered  # Have loader but may not be in S3
    }


def download_dataset(name: str, force: bool = False) -> str:
    """Download dataset to cache and return path"""
    manager = get_dataset_manager()
    return manager.download_dataset(name, force=force)


def register_custom_loader(name: str, loader_func):
    """
    Register a custom dataset loader globally.
    
    Args:
        name: Dataset name
        loader_func: Function(dataset_dir, **kwargs) -> dataset
        
    Example:
        def load_my_dataset(dataset_dir, **kwargs):
            data = torch.load(os.path.join(dataset_dir, 'data.pt'))
            return data
        
        register_custom_loader('my_dataset', load_my_dataset)
        train, test = load_dataset('my_dataset')
    """
    manager = get_dataset_manager()
    manager.register_dataset_loader(name, loader_func)


if __name__ == '__main__':
    # Test the dataset loader
    logging.basicConfig(level=logging.INFO)
    
    manager = DatasetManager()
    
    print("Dataset Status:")
    print("=" * 60)
    status = list_available_datasets()
    print(f"\n✓ Available (ready to use): {len(status['available'])} datasets")
    for ds in status['available']:
        print(f"    - {ds}")
    
    print(f"\n📦 In S3: {len(status['in_s3'])} datasets")
    for ds in status['in_s3']:
        print(f"    - {ds}")
    
    print(f"\n🔧 Registered loaders: {len(status['registered'])} loaders")
    for ds in status['registered']:
        print(f"    - {ds}")
    
    print("\n" + "=" * 60)
    print("Testing CIFAR-10 download and load...")
    train_loader, test_loader = manager.load_dataset('cifar10', batch_size=64)
    print(f"✓ CIFAR-10 loaded: {len(train_loader)} train batches, {len(test_loader)} test batches")
    
    print("\n" + "=" * 60)
    print("Testing custom loader registration...")
    
    def custom_loader(dataset_dir, **kwargs):
        """Example custom loader"""
        return {"data": "custom dataset", "source": dataset_dir}
    
    manager.register_dataset_loader('my_custom_dataset', custom_loader)
    print("✓ Custom loader registered")
    print(f"  Registered datasets now: {list(manager._loader_registry.keys())}")

