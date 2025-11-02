#!/usr/bin/env python3
"""
Test Dataset Pipeline

This script tests the entire dataset pipeline:
1. Verifies S3 bucket access
2. Checks dataset availability
3. Tests dataset downloads
4. Validates dataset integrity
5. Simulates Trainium code execution with datasets
"""

import sys
import boto3
import json
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'trainium_executor'))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

S3_BUCKET = 'datasets-for-all-papers'

def test_s3_access():
    """Test S3 bucket access"""
    logger.info("=" * 60)
    logger.info("TEST 1: S3 Bucket Access")
    logger.info("=" * 60)
    
    try:
        s3_client = boto3.client('s3')
        s3_client.head_bucket(Bucket=S3_BUCKET)
        logger.info(f"✓ S3 bucket '{S3_BUCKET}' is accessible")
        return True
    except Exception as e:
        logger.error(f"✗ Cannot access S3 bucket '{S3_BUCKET}': {e}")
        logger.error("  Make sure the bucket exists and you have proper IAM permissions")
        return False

def test_dataset_index():
    """Test dataset index availability"""
    logger.info("=" * 60)
    logger.info("TEST 2: Dataset Index")
    logger.info("=" * 60)
    
    try:
        s3_client = boto3.client('s3')
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key='dataset_index.json')
        index = json.loads(obj['Body'].read().decode('utf-8'))
        
        logger.info(f"✓ Dataset index found")
        logger.info(f"  Total datasets: {index['total_datasets']}")
        logger.info(f"  Total size: {index['total_size_mb']:.2f} MB")
        logger.info(f"  Last updated: {index['last_updated']}")
        
        logger.info("\n  Available datasets:")
        for ds in index['datasets']:
            logger.info(f"    - {ds['name']}: {ds['size_mb']:.2f} MB ({ds['num_samples']} samples)")
        
        return True, index
    except s3_client.exceptions.NoSuchKey:
        logger.error("✗ Dataset index not found in S3")
        logger.error("  Run 'python upload_datasets_to_s3.py' to create it")
        return False, None
    except Exception as e:
        logger.error(f"✗ Error reading dataset index: {e}")
        return False, None

def test_dataset_presence(dataset_name: str):
    """Test if a specific dataset exists in S3"""
    logger.info(f"  Checking {dataset_name}...")
    
    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=f"{dataset_name}/",
            MaxKeys=1
        )
        
        if 'Contents' in response and len(response['Contents']) > 0:
            logger.info(f"    ✓ {dataset_name} found in S3")
            return True
        else:
            logger.warning(f"    ✗ {dataset_name} not found in S3")
            return False
    except Exception as e:
        logger.error(f"    ✗ Error checking {dataset_name}: {e}")
        return False

def test_all_datasets_present(index):
    """Test that all datasets in index are present in S3"""
    logger.info("=" * 60)
    logger.info("TEST 3: Dataset Presence in S3")
    logger.info("=" * 60)
    
    if not index:
        logger.warning("  Skipping (no index available)")
        return False
    
    all_present = True
    for ds in index['datasets']:
        if not test_dataset_presence(ds['name']):
            all_present = False
    
    if all_present:
        logger.info("\n✓ All datasets are present in S3")
    else:
        logger.warning("\n⚠ Some datasets are missing from S3")
    
    return all_present

def test_dataset_loader_import():
    """Test that dataset_loader module can be imported"""
    logger.info("=" * 60)
    logger.info("TEST 4: Dataset Loader Import")
    logger.info("=" * 60)
    
    try:
        from dataset_loader import DatasetManager, load_dataset, list_available_datasets
        logger.info("✓ dataset_loader module imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import dataset_loader: {e}")
        logger.error("  Make sure dataset_loader.py exists in trainium_executor/")
        return False
    except Exception as e:
        logger.error(f"✗ Error importing dataset_loader: {e}")
        return False

def test_dataset_download():
    """Test downloading a small dataset"""
    logger.info("=" * 60)
    logger.info("TEST 5: Dataset Download")
    logger.info("=" * 60)
    
    try:
        from dataset_loader import DatasetManager
        
        manager = DatasetManager(cache_dir='./test_cache')
        
        # Test with MNIST (smaller dataset)
        logger.info("  Downloading MNIST for testing...")
        dataset_dir = manager.download_dataset('mnist')
        
        if os.path.exists(dataset_dir):
            logger.info(f"✓ MNIST downloaded successfully to {dataset_dir}")
            
            # Check file count
            file_count = sum(1 for _ in Path(dataset_dir).rglob('*') if _.is_file())
            logger.info(f"  Downloaded {file_count} files")
            
            # Cleanup
            import shutil
            shutil.rmtree('./test_cache')
            logger.info("  Cleaned up test cache")
            
            return True
        else:
            logger.error("✗ Dataset directory not created")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test loading and using a dataset"""
    logger.info("=" * 60)
    logger.info("TEST 6: Dataset Loading and Usage")
    logger.info("=" * 60)
    
    try:
        from dataset_loader import DatasetManager
        import torch
        
        manager = DatasetManager(cache_dir='./test_cache')
        
        logger.info("  Loading MNIST dataset...")
        train_loader, test_loader = manager.load_dataset('mnist', batch_size=64)
        
        logger.info(f"  ✓ Train loader: {len(train_loader)} batches")
        logger.info(f"  ✓ Test loader: {len(test_loader)} batches")
        
        # Test iterating through one batch
        logger.info("  Testing batch iteration...")
        images, labels = next(iter(train_loader))
        
        logger.info(f"  ✓ Batch shape: {images.shape}")
        logger.info(f"  ✓ Labels shape: {labels.shape}")
        logger.info(f"  ✓ Data type: {images.dtype}")
        
        # Cleanup
        import shutil
        shutil.rmtree('./test_cache')
        
        logger.info("\n✓ Dataset loading and usage test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_example_training_code():
    """Test example training code with dataset"""
    logger.info("=" * 60)
    logger.info("TEST 7: Example Training Code")
    logger.info("=" * 60)
    
    try:
        from dataset_loader import load_dataset
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(28*28, 10)
            
            def forward(self, x):
                x = self.flatten(x)
                return self.fc(x)
        
        logger.info("  Loading dataset...")
        train_loader, test_loader = load_dataset('mnist', batch_size=64)
        
        logger.info("  Creating model...")
        model = SimpleModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        logger.info("  Running 1 epoch of training...")
        model.train()
        total_loss = 0
        num_batches = min(10, len(train_loader))  # Only train on first 10 batches
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        logger.info(f"  ✓ Average loss: {avg_loss:.4f}")
        
        # Cleanup
        import shutil
        shutil.rmtree('./test_cache')
        
        logger.info("\n✓ Example training code test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error in training code: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_summary(results):
    """Print test summary"""
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Dataset pipeline is ready.")
        logger.info("\nNext steps:")
        logger.info("  1. Deploy to Trainium: ./deployment/deploy_trainium.sh")
        logger.info("  2. Update code generation prompts to use load_dataset()")
        logger.info("  3. Test with generated code on Trainium")
    else:
        logger.warning("\n⚠ Some tests failed. Please fix the issues before deploying.")
        if not results.get('S3 Access'):
            logger.warning("  → Run: python upload_datasets_to_s3.py")
        if not results.get('Dataset Loader Import'):
            logger.warning("  → Check that trainium_executor/dataset_loader.py exists")

def main():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("DATASET PIPELINE TEST SUITE")
    logger.info("=" * 60)
    logger.info(f"Target S3 bucket: {S3_BUCKET}\n")
    
    results = {}
    
    # Test 1: S3 Access
    results['S3 Access'] = test_s3_access()
    if not results['S3 Access']:
        logger.error("\n⚠ Cannot proceed without S3 access. Stopping tests.")
        print_summary(results)
        return
    
    # Test 2: Dataset Index
    index_ok, index = test_dataset_index()
    results['Dataset Index'] = index_ok
    
    # Test 3: Dataset Presence
    results['Dataset Presence'] = test_all_datasets_present(index)
    
    # Test 4: Dataset Loader Import
    results['Dataset Loader Import'] = test_dataset_loader_import()
    if not results['Dataset Loader Import']:
        logger.error("\n⚠ Cannot proceed without dataset_loader. Stopping tests.")
        print_summary(results)
        return
    
    # Test 5: Dataset Download
    results['Dataset Download'] = test_dataset_download()
    
    # Test 6: Dataset Loading
    results['Dataset Loading'] = test_dataset_loading()
    
    # Test 7: Example Training Code
    results['Example Training Code'] = test_example_training_code()
    
    # Print summary
    print_summary(results)

if __name__ == '__main__':
    main()

