#!/usr/bin/env python3
"""
Dataset Loader for Trainium Executor

This module provides utilities to download and load datasets from S3
for use by generated PyTorch code on Trainium instances.
"""

import os
import boto3
import torch
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

S3_BUCKET = 'datasets-for-all-papers'
LOCAL_DATA_DIR = os.getenv('DATASET_CACHE_DIR', '/tmp/datasets')
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

s3_client = boto3.client('s3')

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
            'imdb': self._load_imdb,
            'wikitext2': self._load_wikitext,
            'synthetic': self._load_synthetic
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
        """Load CIFAR-10 dataset"""
        from torchvision.datasets import CIFAR10
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = CIFAR10(root=dataset_dir, train=True, transform=transform)
        test_dataset = CIFAR10(root=dataset_dir, train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    
    def _load_cifar100(self, dataset_dir: str, batch_size: int = 128, **kwargs):
        """Load CIFAR-100 dataset"""
        from torchvision.datasets import CIFAR100
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = CIFAR100(root=dataset_dir, train=True, transform=transform)
        test_dataset = CIFAR100(root=dataset_dir, train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    
    def _load_mnist(self, dataset_dir: str, batch_size: int = 128, **kwargs):
        """Load MNIST dataset"""
        from torchvision.datasets import MNIST
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = MNIST(root=dataset_dir, train=True, transform=transform)
        test_dataset = MNIST(root=dataset_dir, train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    
    def _load_fashion_mnist(self, dataset_dir: str, batch_size: int = 128, **kwargs):
        """Load Fashion-MNIST dataset"""
        from torchvision.datasets import FashionMNIST
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = FashionMNIST(root=dataset_dir, train=True, transform=transform)
        test_dataset = FashionMNIST(root=dataset_dir, train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    
    def _load_imdb(self, dataset_dir: str, **kwargs):
        """Load IMDB dataset"""
        from datasets import load_from_disk
        
        dataset = load_from_disk(dataset_dir)
        return dataset['train'], dataset['test']
    
    def _load_wikitext(self, dataset_dir: str, **kwargs):
        """Load WikiText-2 dataset"""
        from datasets import load_from_disk
        
        dataset = load_from_disk(dataset_dir)
        return dataset
    
    def _load_synthetic(self, dataset_dir: str, variant: str = 'small', **kwargs):
        """Load synthetic dataset"""
        filename = f"synthetic_{variant}.pt"
        filepath = os.path.join(dataset_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Synthetic variant '{variant}' not found. Available: small, medium, tabular")
        
        return torch.load(filepath)


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

