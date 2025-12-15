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
    
    def __init__(self, cache_dir: str = LOCAL_DATA_DIR):
        self.cache_dir = cache_dir
        self.s3_bucket = S3_BUCKET
        os.makedirs(self.cache_dir, exist_ok=True)
        self._index = None
        self._loader_registry = self._build_loader_registry()
    
    def _build_loader_registry(self):
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
        self._loader_registry[name] = loader_func
        logger.info(f"Registered custom loader for dataset: {name}")
    
    def get_dataset_index(self) -> Dict[str, Any]:
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
        index = self.get_dataset_index()
        index_datasets = [ds['name'] for ds in index.get('datasets', [])]
        
        try:
            response = s3_client.list_objects_v2(Bucket=self.s3_bucket, Delimiter='/')
            s3_datasets = [obj['Prefix'].rstrip('/') for obj in response.get('CommonPrefixes', [])]
            
            all_datasets = set(index_datasets) | set(s3_datasets)
            available = [ds for ds in all_datasets if ds in self._loader_registry]
            return sorted(available)
        except Exception as e:
            logger.warning(f"Failed to list datasets from S3: {e}. Using index only.")
            return index_datasets
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> str:
        dataset_dir = os.path.join(self.cache_dir, dataset_name)
        
        if os.path.exists(dataset_dir) and not force:
            logger.info(f"Dataset '{dataset_name}' already cached at {dataset_dir}")
            return dataset_dir
        
        os.makedirs(dataset_dir, exist_ok=True)
        
        logger.info(f"Downloading dataset '{dataset_name}' from S3...")
        
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=f"{dataset_name}/")
            
            file_count = 0
            for page in pages:
                for obj in page.get('Contents', []):
                    s3_key = obj['Key']
                    
                    if s3_key == f"{dataset_name}/":
                        continue
                    
                    relative_path = s3_key[len(dataset_name)+1:]  # Remove prefix
                    local_file = os.path.join(dataset_dir, relative_path)
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    
                    s3_client.download_file(self.s3_bucket, s3_key, local_file)
                    file_count += 1
            
            logger.info(f"✓ Downloaded {file_count} files for '{dataset_name}' to {dataset_dir}")
            return dataset_dir
            
        except Exception as e:
            logger.error(f"Failed to download dataset '{dataset_name}': {e}")
            raise
    
    def load_dataset(self, dataset_name: str, **kwargs):
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
    
    # Dataset loader functions:
    
    def _load_cifar10(self, dataset_dir: str, batch_size: int = 128, **kwargs):
        pt_file = os.path.join(dataset_dir, 'cifar10_pytorch.pt')
        
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"CIFAR-10 data file not found: {pt_file}. Please download from S3 first.")
        
        data = torch.load(pt_file, map_location='cpu')
        
        try:
            if isinstance(data, dict) and 'train_data' in data:
                train_images = data['train_data']
                train_labels = data['train_labels']
                test_images = data['test_data']
                test_labels = data['test_labels']
                logger.info(f"Loaded CIFAR-10 from tensor format: train={train_images.shape}, test={test_images.shape}")
            elif isinstance(data, dict) and 'train' in data and 'test' in data:
                train_dataset_obj = data.get('train')
                test_dataset_obj = data.get('test')
                
                if hasattr(train_dataset_obj, '__getitem__') and hasattr(train_dataset_obj, '__len__'):
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
                    raise ValueError(f"CIFAR-10 'train'/'test' objects don't have expected dataset interface. Type: {type(train_dataset_obj)}")
            else:
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
        
        def normalize_transform(img):
            if img.max() > 1.0:
                img = img.float() / 255.0
            return (img - 0.5) / 0.5
        
        train_dataset = SimpleImageDataset(train_images, train_labels, transform=normalize_transform)
        test_dataset = SimpleImageDataset(test_images, test_labels, transform=normalize_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    
    def _load_cifar100(self, dataset_dir: str, batch_size: int = 128, **kwargs):
        # Use the correct filename as confirmed in S3
        pt_file = os.path.join(dataset_dir, 'cifar100_pytorch.pt')
        
        if not os.path.exists(pt_file):
            raise FileNotFoundError(
                f"CIFAR-100 data file not found: {pt_file}. "
                f"Please download from S3 first."
            )
        
        logger.info(f"Loading CIFAR-100 from {pt_file}")
        data = torch.load(pt_file, map_location='cpu')
        
        try:
            if isinstance(data, dict) and 'train_data' in data:
                train_images = data['train_data']
                train_labels = data['train_labels']
                test_images = data['test_data']
                test_labels = data['test_labels']
                logger.info(f"Loaded CIFAR-100 from tensor format: train={train_images.shape}, test={test_images.shape}")
            elif isinstance(data, dict) and 'train' in data and 'test' in data:
                train_dataset_obj = data.get('train')
                test_dataset_obj = data.get('test')
                
                if hasattr(train_dataset_obj, '__getitem__') and hasattr(train_dataset_obj, '__len__'):
                    logger.info(f"Extracting CIFAR-100 from dataset objects: train={len(train_dataset_obj)}, test={len(test_dataset_obj)}")
                    batch_size_extract = 1000
                    train_batches = []
                    train_label_batches = []
                    for i in range(0, len(train_dataset_obj), batch_size_extract):
                        batch_end = min(i + batch_size_extract, len(train_dataset_obj))
                        batch_data = [train_dataset_obj[j] for j in range(i, batch_end)]
                        train_batches.append(torch.stack([torch.tensor(item[0]) for item in batch_data]))
                        train_label_batches.append(torch.tensor([item[1] for item in batch_data]))
                    train_images = torch.cat(train_batches, dim=0)
                    train_labels = torch.cat(train_label_batches, dim=0)
                    
                    test_batches = []
                    test_label_batches = []
                    for i in range(0, len(test_dataset_obj), batch_size_extract):
                        batch_end = min(i + batch_size_extract, len(test_dataset_obj))
                        batch_data = [test_dataset_obj[j] for j in range(i, batch_end)]
                        test_batches.append(torch.stack([torch.tensor(item[0]) for item in batch_data]))
                        test_label_batches.append(torch.tensor([item[1] for item in batch_data]))
                    test_images = torch.cat(test_batches, dim=0)
                    test_labels = torch.cat(test_label_batches, dim=0)
                    logger.info(f"Extracted CIFAR-100: train={train_images.shape}, test={test_images.shape}")
                else:
                    raise ValueError(f"CIFAR-100 'train'/'test' objects don't have expected dataset interface. Type: {type(train_dataset_obj)}")
            else:
                data_info = f"Type: {type(data)}"
                if isinstance(data, dict):
                    data_info += f", Keys: {list(data.keys())}"
                    data_info += f", Sample values: {[(k, type(v).__name__) for k, v in list(data.items())[:5]]}"
                raise ValueError(
                    f"Unknown CIFAR-100 data format in {pt_file}. "
                    f"Expected dict with 'train_data'/'train_labels' or 'train'/'test' keys. "
                    f"Got: {data_info}"
                )
        except Exception as e:
            logger.error(f"Error extracting CIFAR-100 data from {pt_file}: {e}")
            logger.error(f"Data type: {type(data)}, Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Validate shapes - CIFAR-100 should be (N, 3, 32, 32) or (N, 3072) if flattened
        if len(train_images.shape) == 2:
            if train_images.shape[1] == 3072: 
                train_images = train_images.view(-1, 3, 32, 32)
                test_images = test_images.view(-1, 3, 32, 32)
                logger.info(f"Reshaped flattened CIFAR-100 to: train={train_images.shape}, test={test_images.shape}")
            else:
                logger.warning(f"Unexpected flattened shape: {train_images.shape}. Expected (N, 3072)")
        
        if len(train_images.shape) != 4 or train_images.shape[1] != 3 or train_images.shape[2] != 32 or train_images.shape[3] != 32:
            logger.warning(f"CIFAR-100 images have unexpected shape: {train_images.shape}. Expected (N, 3, 32, 32)")
        
        def normalize_transform(img):
            if img.max() > 1.0:
                img = img.float() / 255.0
            return (img - 0.5) / 0.5

        train_dataset = SimpleImageDataset(train_images, train_labels, transform=normalize_transform)
        test_dataset = SimpleImageDataset(test_images, test_labels, transform=normalize_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        logger.info(f"CIFAR-100 loaded successfully: {len(train_loader)} train batches, {len(test_loader)} test batches")
        return train_loader, test_loader
    
    def _load_mnist(self, dataset_dir: str, batch_size: int = 128, **kwargs):
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
        try:
            from datasets import load_from_disk
        except Exception as e:
            raise ImportError(
                f"HuggingFace 'datasets' package is required for IMDB but failed to import: {e}. "
                "Please use a different dataset (e.g., mnist, cifar10, cifar100, fashion_mnist)."
            )
        
        dataset = load_from_disk(dataset_dir)
        
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        
        class IMDBDataset(Dataset):
            def __init__(self, hf_dataset):
                self.hf_dataset = hf_dataset
            
            def __len__(self):
                return len(self.hf_dataset)
            
            def __getitem__(self, idx):
                item = self.hf_dataset[idx]
                return item['text'], item['label']
        
        train_pytorch_dataset = IMDBDataset(train_dataset)
        test_pytorch_dataset = IMDBDataset(test_dataset)
        
        train_loader = DataLoader(train_pytorch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_pytorch_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    
    def _load_wikitext(self, dataset_dir: str, batch_size: int = 128, seq_length: int = 128, max_vocab_size: int = 10000, **kwargs):
        try:
            from datasets import load_from_disk
            from collections import Counter
        except Exception as e:
            raise ImportError(
                f"HuggingFace 'datasets' package is required for WikiText-2 but failed to import: {e}. "
                "Please use a different dataset (e.g., mnist, cifar10, cifar100, fashion_mnist)."
            )
        
        dataset = load_from_disk(dataset_dir)
        
        train_dataset = dataset.get('train', dataset.get('validation'))
        val_dataset = dataset.get('validation', dataset.get('test'))
        
        logger.info("Building vocabulary from WikiText-2 (memory-efficient)...")
        word_counts = Counter()
        
        sample_size = min(10000, len(train_dataset))
        for i in range(sample_size):
            item = train_dataset[i]
            text = item.get('text', '').strip()
            if len(text) > 0:
                words = text.lower().split()
                word_counts.update(words)
        
        vocab = {'<pad>': 0, '<unk>': 1}
        vocab_size = 2
        for word, count in word_counts.most_common(max_vocab_size - 2):
            vocab[word] = vocab_size
            vocab_size += 1
        
        logger.info(f"Built vocabulary with {len(vocab)} words (limited to {max_vocab_size})")
        
        class WikiTextDataset(Dataset):
            def __init__(self, hf_dataset, vocab, seq_length=128):
                self.hf_dataset = hf_dataset
                self.vocab = vocab
                self.seq_length = seq_length
                self.vocab_size = len(vocab)
                self.valid_indices = []
                for i in range(len(hf_dataset)):
                    item = hf_dataset[i]
                    text = item.get('text', '').strip()
                    if len(text) > 0:
                        words = text.lower().split()
                        if len(words) >= seq_length:
                            self.valid_indices.append(i)
            
            def __len__(self):
                return min(len(self.valid_indices) * 10, 50000)
            
            def __getitem__(self, idx):
                actual_idx = self.valid_indices[idx % len(self.valid_indices)]
                item = self.hf_dataset[actual_idx]
                text = item.get('text', '').strip()
                
                words = text.lower().split()
                token_ids = [self.vocab.get(word, 1) for word in words] 
                
                offset = (idx * 7) % max(1, len(token_ids) - self.seq_length) 
                start_idx = min(offset, len(token_ids) - self.seq_length)
                input_ids = token_ids[start_idx:start_idx + self.seq_length]
                
                if len(input_ids) < self.seq_length:
                    input_ids = input_ids + [0] * (self.seq_length - len(input_ids))
                else:
                    input_ids = input_ids[:self.seq_length]
                
                labels = input_ids[1:] + [0] 
                
                input_tensor = torch.tensor(input_ids, dtype=torch.long)
                label_tensor = torch.tensor(labels, dtype=torch.long)
                
                return input_tensor, label_tensor
        
        train_pytorch_dataset = WikiTextDataset(train_dataset, vocab, seq_length=seq_length)
        val_pytorch_dataset = WikiTextDataset(val_dataset, vocab, seq_length=seq_length)
        
        train_loader = DataLoader(train_pytorch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(val_pytorch_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    
    def _load_synthetic(self, dataset_dir: str, batch_size: int = 128, variant: str = 'small', **kwargs):
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
        
        data = torch.load(pt_file, map_location='cpu')
        
        if 'images' in data and 'labels' in data:
            images = data['images']
            labels = data['labels']
            
            def normalize_transform(img):
                if img.max() > 1.0:
                    img = img.float() / 255.0
                return (img - 0.5) / 0.5
            
            train_dataset = SimpleImageDataset(images, labels, transform=normalize_transform)
            test_dataset = SimpleImageDataset(images, labels, transform=normalize_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            return train_loader, test_loader
        
        elif 'features' in data and 'labels' in data:
            features = data['features']
            labels = data['labels']
            
            class TabularDataset(Dataset):
                def __init__(self, features, labels):
                    self.features = features
                    self.labels = labels
                
                def __len__(self):
                    return len(self.labels)
                
                def __getitem__(self, idx):
                    return self.features[idx], self.labels[idx]
            
            train_dataset = TabularDataset(features, labels)
            test_dataset = TabularDataset(features, labels)  
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            return train_loader, test_loader
        
        else:
            raise ValueError(f"Unknown synthetic dataset format in {pt_file}")


_dataset_manager = None

def get_dataset_manager() -> DatasetManager:
    global _dataset_manager
    if _dataset_manager is None:
        _dataset_manager = DatasetManager()
    return _dataset_manager


def load_dataset(name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    manager = get_dataset_manager()
    return manager.load_dataset(name, **kwargs)


def list_available_datasets():
    """List all available datasets"""
    manager = get_dataset_manager()
    
    s3_datasets = manager.list_datasets()
    
    registered = list(manager._loader_registry.keys())
    
    available = [ds for ds in s3_datasets if ds in registered]
    
    return {
        'available': available,  
        'in_s3': s3_datasets,   
        'registered': registered 
    }


def download_dataset(name: str, force: bool = False) -> str:
    manager = get_dataset_manager()
    return manager.download_dataset(name, force=force)


def register_custom_loader(name: str, loader_func):
    manager = get_dataset_manager()
    manager.register_dataset_loader(name, loader_func)


if __name__ == '__main__':
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



