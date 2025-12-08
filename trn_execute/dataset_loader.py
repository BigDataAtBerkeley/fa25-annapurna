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
            'wikitext2': self._load_wikitext,  # Uses HuggingFace datasets.load_from_disk
            'synthetic': self._load_synthetic  # Uses pre-processed .pt files
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
        """List all available datasets from S3 (not just index)"""
        # Try to get from index first
        index = self.get_dataset_index()
        index_datasets = [ds['name'] for ds in index.get('datasets', [])]
        
        # Also check S3 directly for datasets with registered loaders
        # This ensures we find datasets even if index is outdated
        try:
            response = s3_client.list_objects_v2(Bucket=self.s3_bucket, Delimiter='/')
            s3_datasets = [obj['Prefix'].rstrip('/') for obj in response.get('CommonPrefixes', [])]
            
            # Return union of index datasets and S3 datasets that have loaders
            all_datasets = set(index_datasets) | set(s3_datasets)
            # Filter to only datasets with registered loaders
            available = [ds for ds in all_datasets if ds in self._loader_registry]
            return sorted(available)
        except Exception as e:
            logger.warning(f"Failed to list datasets from S3: {e}. Using index only.")
            return index_datasets
    
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
            Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
            ALWAYS returns exactly 2 DataLoaders. Do NOT attempt to unpack 3 values.
            
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
                logger.info(f"Loaded CIFAR-10 from tensor format: train={train_images.shape}, test={test_images.shape}")
            elif isinstance(data, dict) and 'train' in data and 'test' in data:
                # Extract from dataset objects (more memory-efficient batch processing)
                train_dataset_obj = data.get('train')
                test_dataset_obj = data.get('test')
                
                # Check if dataset objects are already tensors or need extraction
                if hasattr(train_dataset_obj, '__getitem__') and hasattr(train_dataset_obj, '__len__'):
                    # Extract in batches to avoid OOM
                    logger.info(f"Extracting CIFAR-10 from dataset objects: train={len(train_dataset_obj)}, test={len(test_dataset_obj)}")
                    batch_size = 1000
                    train_batches = []
                    train_label_batches = []
                    for i in range(0, len(train_dataset_obj), batch_size):
                        batch_end = min(i + batch_size, len(train_dataset_obj))
                        batch_data = [train_dataset_obj[j] for j in range(i, batch_end)]
                        train_batches.append(torch.stack([torch.tensor(item[0]) for item in batch_data]))
                        train_label_batches.append(torch.tensor([item[1] for item in batch_data]))
                    train_images = torch.cat(train_batches, dim=0)
                    train_labels = torch.cat(train_label_batches, dim=0)
                    
                    # Same for test set
                    test_batches = []
                    test_label_batches = []
                    for i in range(0, len(test_dataset_obj), batch_size):
                        batch_end = min(i + batch_size, len(test_dataset_obj))
                        batch_data = [test_dataset_obj[j] for j in range(i, batch_end)]
                        test_batches.append(torch.stack([torch.tensor(item[0]) for item in batch_data]))
                        test_label_batches.append(torch.tensor([item[1] for item in batch_data]))
                    test_images = torch.cat(test_batches, dim=0)
                    test_labels = torch.cat(test_label_batches, dim=0)
                    logger.info(f"Extracted CIFAR-10: train={train_images.shape}, test={test_images.shape}")
                else:
                    # Already tensors or unexpected format
                    raise ValueError(f"CIFAR-10 'train'/'test' objects don't have expected dataset interface. Type: {type(train_dataset_obj)}")
            else:
                # Log detailed error information
                data_info = f"Type: {type(data)}"
                if isinstance(data, dict):
                    data_info += f", Keys: {list(data.keys())}"
                    data_info += f", Sample values: {[(k, type(v).__name__) for k, v in list(data.items())[:5]]}"
                raise ValueError(
                    f"Unknown CIFAR-10 data format in {pt_file}. "
                    f"Expected dict with 'train_data'/'train_labels' or 'train'/'test' keys. "
                    f"Got: {data_info}"
                )
        except Exception as e:
            logger.error(f"Error extracting CIFAR-10 data from {pt_file}: {e}")
            logger.error(f"Data type: {type(data)}, Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            import traceback
            logger.error(traceback.format_exc())
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
    
    def _load_wikitext(self, dataset_dir: str, batch_size: int = 128, seq_length: int = 128, max_vocab_size: int = 10000, **kwargs):
        """Load WikiText-2 dataset from HuggingFace Arrow format for language modeling.
        
        Memory-efficient implementation:
        - Limits vocabulary size to avoid OOM
        - Uses lazy loading (no pre-computed sequences)
        - Smaller default seq_length (128 instead of 512)
        
        Returns tokenized sequences where:
        - input_ids: tokenized sequence [0, 1, 2, ..., n-1]
        - labels: shifted sequence [1, 2, 3, ..., n] for next-token prediction
        """
        try:
            from datasets import load_from_disk
            from collections import Counter
        except Exception as e:
            raise ImportError(
                f"HuggingFace 'datasets' package is required for WikiText-2 but failed to import: {e}. "
                "Please use a different dataset (e.g., mnist, cifar10, cifar100, fashion_mnist)."
            )
        
        # Load from Arrow format (as stored in S3)
        dataset = load_from_disk(dataset_dir)
        
        # WikiText-2 has 'text' column
        train_dataset = dataset.get('train', dataset.get('validation'))
        val_dataset = dataset.get('validation', dataset.get('test'))
        
        # Build vocabulary efficiently - only count words, don't store all texts
        logger.info("Building vocabulary from WikiText-2 (memory-efficient)...")
        word_counts = Counter()
        
        # Sample first 10k examples to build vocab (faster, less memory)
        sample_size = min(10000, len(train_dataset))
        for i in range(sample_size):
            item = train_dataset[i]
            text = item.get('text', '').strip()
            if len(text) > 0:
                words = text.lower().split()
                word_counts.update(words)
        
        # Build vocab from most frequent words (limit to max_vocab_size)
        vocab = {'<pad>': 0, '<unk>': 1}
        vocab_size = 2
        for word, count in word_counts.most_common(max_vocab_size - 2):
            vocab[word] = vocab_size
            vocab_size += 1
        
        logger.info(f"Built vocabulary with {len(vocab)} words (limited to {max_vocab_size})")
        
        # Create memory-efficient dataset wrapper with lazy loading
        class WikiTextDataset(Dataset):
            def __init__(self, hf_dataset, vocab, seq_length=128):
                self.hf_dataset = hf_dataset
                self.vocab = vocab
                self.seq_length = seq_length
                self.vocab_size = len(vocab)
                # Don't pre-compute sequences - calculate on-the-fly
                # Just store indices of valid items
                self.valid_indices = []
                for i in range(len(hf_dataset)):
                    item = hf_dataset[i]
                    text = item.get('text', '').strip()
                    if len(text) > 0:
                        words = text.lower().split()
                        if len(words) >= seq_length:
                            self.valid_indices.append(i)
            
            def __len__(self):
                # Estimate: each valid item can produce multiple sequences
                # Return a reasonable number for training
                return min(len(self.valid_indices) * 10, 50000)  # Cap at 50k sequences
            
            def __getitem__(self, idx):
                # Map idx to actual dataset item (with some randomness for variety)
                actual_idx = self.valid_indices[idx % len(self.valid_indices)]
                item = self.hf_dataset[actual_idx]
                text = item.get('text', '').strip()
                
                # Tokenize on-the-fly
                words = text.lower().split()
                token_ids = [self.vocab.get(word, 1) for word in words]  # 1 = <unk>
                
                # Extract a sequence of seq_length (with some offset for variety)
                offset = (idx * 7) % max(1, len(token_ids) - self.seq_length)  # Simple pseudo-random offset
                start_idx = min(offset, len(token_ids) - self.seq_length)
                input_ids = token_ids[start_idx:start_idx + self.seq_length]
                
                # Ensure we have exactly seq_length tokens
                if len(input_ids) < self.seq_length:
                    # Pad if needed
                    input_ids = input_ids + [0] * (self.seq_length - len(input_ids))
                else:
                    input_ids = input_ids[:self.seq_length]
                
                # Labels are input_ids shifted by 1 for next-token prediction
                labels = input_ids[1:] + [0]  # Last token predicts padding
                
                # Convert to tensors
                input_tensor = torch.tensor(input_ids, dtype=torch.long)
                label_tensor = torch.tensor(labels, dtype=torch.long)
                
                return input_tensor, label_tensor
        
        train_pytorch_dataset = WikiTextDataset(train_dataset, vocab, seq_length=seq_length)
        val_pytorch_dataset = WikiTextDataset(val_dataset, vocab, seq_length=seq_length)
        
        train_loader = DataLoader(train_pytorch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(val_pytorch_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    
    def _load_synthetic(self, dataset_dir: str, batch_size: int = 128, variant: str = 'small', **kwargs):
        """Load synthetic dataset from pre-processed .pt file"""
        # Determine which synthetic dataset file to load
        if variant == 'small':
            pt_file = os.path.join(dataset_dir, 'synthetic_small.pt')
        elif variant == 'medium':
            pt_file = os.path.join(dataset_dir, 'synthetic_medium.pt')
        elif variant == 'tabular':
            pt_file = os.path.join(dataset_dir, 'synthetic_tabular.pt')
        else:
            pt_file = os.path.join(dataset_dir, 'synthetic_small.pt')  # Default to small
        
        if not os.path.exists(pt_file):
            raise FileNotFoundError(
                f"Synthetic dataset file not found: {pt_file}. "
                f"Available variants: small, medium, tabular. "
                f"Please download from S3 first."
            )
        
        # Load pre-processed data
        data = torch.load(pt_file, map_location='cpu')
        
        # Handle different synthetic dataset formats
        if 'images' in data and 'labels' in data:
            # Vision-style synthetic data
            images = data['images']
            labels = data['labels']
            
            # Normalize images if needed
            def normalize_transform(img):
                if img.max() > 1.0:
                    img = img.float() / 255.0
                return (img - 0.5) / 0.5
            
            train_dataset = SimpleImageDataset(images, labels, transform=normalize_transform)
            # For synthetic, use same data for train and test
            test_dataset = SimpleImageDataset(images, labels, transform=normalize_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            return train_loader, test_loader
        
        elif 'features' in data and 'labels' in data:
            # Tabular synthetic data
            features = data['features']
            labels = data['labels']
            
            # Create simple tabular dataset
            class TabularDataset(Dataset):
                def __init__(self, features, labels):
                    self.features = features
                    self.labels = labels
                
                def __len__(self):
                    return len(self.labels)
                
                def __getitem__(self, idx):
                    return self.features[idx], self.labels[idx]
            
            train_dataset = TabularDataset(features, labels)
            test_dataset = TabularDataset(features, labels)  # Same data for test
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            return train_loader, test_loader
        
        else:
            raise ValueError(f"Unknown synthetic dataset format in {pt_file}")


# Global dataset manager instance
_dataset_manager = None

def get_dataset_manager() -> DatasetManager:
    """Get or create the global dataset manager"""
    global _dataset_manager
    if _dataset_manager is None:
        _dataset_manager = DatasetManager()
    return _dataset_manager


# Convenience functions for generated code
def load_dataset(name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to load a dataset.
    
    Args:
        name: Dataset name (e.g., 'cifar10', 'mnist', 'imdb')
        **kwargs: Additional arguments (e.g., batch_size=128)
    
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
        ALWAYS returns exactly 2 DataLoaders. Do NOT attempt to unpack 3 values.
    
    Usage in generated code:
        from dataset_loader import load_dataset
        train_loader, test_loader = load_dataset('cifar10', batch_size=128)
        # CRITICAL: Only unpack 2 values, not 3!
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

