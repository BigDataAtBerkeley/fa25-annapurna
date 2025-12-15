#!/usr/bin/env python3
"""
Test script for dataset_loader.py
Tests all datasets to confirm they can be loaded correctly on Trn1.2 instance.
Includes visual confirmations from 1-2 samples of each dataset.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

try:
    from dataset_loader import load_dataset, list_available_datasets
except ImportError as e:
    print(f"❌ Error importing dataset_loader: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available - image display will be limited to tensor info")

# List of all datasets to test
ALL_DATASETS = [
    'cifar10',
    'cifar100',
    'mnist',
    'fashion_mnist',
    'imdb',
    'wikitext2',
    'synthetic'
]

# Image datasets (will display images)
IMAGE_DATASETS = ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'synthetic']

# Text datasets (will display text)
TEXT_DATASETS = ['imdb', 'wikitext2']


def display_image_sample(images, labels, dataset_name, num_samples=2):
    """Display sample images from a dataset"""
    if not HAS_MATPLOTLIB:
        print(f"   Image tensor shape: {images.shape}")
        print(f"   Label tensor shape: {labels.shape}")
        print(f"   Image dtype: {images.dtype}, min: {images.min():.3f}, max: {images.max():.3f}")
        return
    
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        img = images[i]
        label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
        
        # Handle different image formats
        if len(img.shape) == 3:
            # CHW format - convert to HWC for display
            if img.shape[0] == 1:
                # Grayscale
                img_display = img[0].cpu().numpy()
                axes[i].imshow(img_display, cmap='gray')
            elif img.shape[0] == 3:
                # RGB
                img_display = img.permute(1, 2, 0).cpu().numpy()
                # Denormalize if needed (assuming normalized to [-1, 1] or [0, 1])
                if img_display.min() < 0:
                    img_display = (img_display + 1) / 2
                img_display = np.clip(img_display, 0, 1)
                axes[i].imshow(img_display)
            else:
                # Unknown format - show first channel
                img_display = img[0].cpu().numpy()
                axes[i].imshow(img_display, cmap='gray')
        elif len(img.shape) == 2:
            # HW format (grayscale)
            img_display = img.cpu().numpy()
            axes[i].imshow(img_display, cmap='gray')
        else:
            # Flatten or unknown - try to reshape
            print(f"   ⚠️  Unexpected image shape: {img.shape}, skipping display")
            continue
        
        axes[i].set_title(f'{dataset_name}\nLabel: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'test_{dataset_name}_samples.png', dpi=100, bbox_inches='tight')
    print(f"   💾 Saved sample images to: test_{dataset_name}_samples.png")
    plt.close()


def display_text_sample(texts, labels, dataset_name, num_samples=2):
    """Display sample text from a dataset"""
    num_samples = min(num_samples, len(texts))
    
    for i in range(num_samples):
        text = texts[i]
        
        # Handle labels - can be scalars (classification) or sequences (language modeling)
        if torch.is_tensor(labels[i]):
            label_tensor = labels[i]
            if label_tensor.numel() == 1:
                # Scalar label (classification)
                label = label_tensor.item()
            else:
                # Sequence label (language modeling - e.g., wikitext2)
                label = f"[Sequence of {label_tensor.numel()} token IDs: {label_tensor.tolist()[:10]}...]"
        else:
            label = labels[i]
        
        # Handle different text formats
        if isinstance(text, str):
            text_display = text
        elif isinstance(text, torch.Tensor):
            # Tokenized text - convert to readable format
            if text.dim() == 1:
                text_display = f"[Token IDs: {text.tolist()[:20]}...]"  # Show first 20 tokens
            else:
                text_display = f"[Tensor shape: {text.shape}]"
        else:
            text_display = str(text)
        
        # Truncate long text
        if len(text_display) > 200:
            text_display = text_display[:200] + "..."
        
        print(f"   Sample {i+1}:")
        print(f"      Label: {label}")
        print(f"      Text: {text_display}")


def test_dataset(dataset_name, batch_size=32):
    """Test loading a single dataset"""
    print(f"\n{'='*70}")
    print(f"Testing dataset: {dataset_name}")
    print(f"{'='*70}")
    
    try:
        # Load dataset using the recommended method
        print(f"📥 Loading dataset using: from dataset_loader import load_dataset")
        print(f"   Command: train_loader, test_loader = load_dataset('{dataset_name}', batch_size={batch_size})")
        print(f"   Note: load_dataset() automatically downloads from S3 if not cached")
        
        # Check if dataset directory exists before loading
        from dataset_loader import get_dataset_manager
        manager = get_dataset_manager()
        dataset_dir = os.path.join(manager.cache_dir, dataset_name)
        
        if not os.path.exists(dataset_dir):
            print(f"   ℹ️  Dataset not cached - will download from S3 bucket: {manager.s3_bucket}")
        else:
            print(f"   ℹ️  Dataset directory exists: {dataset_dir}")
            # List files in directory
            if os.path.exists(dataset_dir):
                files = os.listdir(dataset_dir)
                print(f"   ℹ️  Cached files: {files[:5]}{'...' if len(files) > 5 else ''}")
        
        train_loader, test_loader = load_dataset(dataset_name, batch_size=batch_size)
        
        # Verify return format
        print(f"✅ Dataset loaded successfully!")
        print(f"   Return type: {type(train_loader).__name__}, {type(test_loader).__name__}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Get a sample batch
        print(f"\n📊 Sample batch analysis:")
        train_batch = next(iter(train_loader))
        
        # Verify batch structure
        if isinstance(train_batch, (list, tuple)) and len(train_batch) == 2:
            inputs, labels = train_batch
            print(f"   Batch structure: (inputs, labels)")
            print(f"   Inputs shape: {inputs.shape if torch.is_tensor(inputs) else type(inputs)}")
            print(f"   Labels shape: {labels.shape if torch.is_tensor(labels) else type(labels)}")
            print(f"   Inputs dtype: {inputs.dtype if torch.is_tensor(inputs) else 'N/A'}")
            print(f"   Labels dtype: {labels.dtype if torch.is_tensor(labels) else 'N/A'}")
            
            # Display samples based on dataset type
            print(f"\n🖼️  Visual confirmation:")
            if dataset_name in IMAGE_DATASETS:
                display_image_sample(inputs[:2], labels[:2], dataset_name, num_samples=2)
            elif dataset_name in TEXT_DATASETS:
                display_text_sample(inputs[:2], labels[:2], dataset_name, num_samples=2)
            else:
                # Unknown type - try to display anyway
                if torch.is_tensor(inputs) and len(inputs.shape) >= 2:
                    # Might be image data
                    display_image_sample(inputs[:2], labels[:2], dataset_name, num_samples=2)
                else:
                    # Text or other
                    display_text_sample(inputs[:2], labels[:2], dataset_name, num_samples=2)
        else:
            print(f"   ⚠️  Unexpected batch structure: {type(train_batch)}")
            print(f"   Batch: {train_batch}")
        
        # Test iteration
        print(f"\n🔄 Testing iteration:")
        sample_count = 0
        for batch_idx, batch in enumerate(train_loader):
            sample_count += 1
            if batch_idx >= 2:  # Test first 3 batches
                break
        print(f"   ✅ Successfully iterated through {sample_count} batches")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Error loading dataset '{dataset_name}': {e}")
        print(f"\n   🔍 Diagnostic information:")
        from dataset_loader import get_dataset_manager
        manager = get_dataset_manager()
        dataset_dir = os.path.join(manager.cache_dir, dataset_name)
        print(f"   - Expected dataset directory: {dataset_dir}")
        print(f"   - Directory exists: {os.path.exists(dataset_dir)}")
        if os.path.exists(dataset_dir):
            files = os.listdir(dataset_dir)
            print(f"   - Files in directory: {files}")
        print(f"   - S3 bucket: {manager.s3_bucket}")
        print(f"   - S3 prefix: {dataset_name}/")
        print(f"\n   💡 This error usually means:")
        print(f"      1. The dataset hasn't been uploaded to S3 bucket '{manager.s3_bucket}'")
        print(f"      2. The S3 file structure doesn't match expected format")
        print(f"      3. The download failed (check AWS credentials and permissions)")
        import traceback
        print(traceback.format_exc())
        return False
    except Exception as e:
        print(f"❌ Error loading dataset '{dataset_name}': {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    """Main test function"""
    print("="*70)
    print("Dataset Loader Test Script")
    print("Testing all datasets for Trn1.2 instance compatibility")
    print("="*70)
    
    # Check available datasets
    print("\n📋 Checking available datasets...")
    try:
        status = list_available_datasets()
        print(f"   Available datasets: {status['available']}")
        print(f"   Registered loaders: {status['registered']}")
    except Exception as e:
        print(f"   ⚠️  Could not check available datasets: {e}")
    
    # Test each dataset
    results = {}
    for dataset_name in ALL_DATASETS:
        success = test_dataset(dataset_name, batch_size=32)
        results[dataset_name] = success
    
    # Summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    print(f"\nDetailed results:")
    for dataset_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {dataset_name}")
    
    if passed == total:
        print(f"\n🎉 All datasets loaded successfully!")
        return 0
    else:
        print(f"\n⚠️  Some datasets failed to load. Check errors above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

