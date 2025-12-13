#!/usr/bin/env python3
"""
Test script for Code Reviewer 0

Tests Code Reviewer 0 locally without deploying to TRN.
Saves all outputs to code_reviewer_0_testing/ folder.

Usage:
    python test_code_reviewer_0.py --code-file path/to/code.py --paper-id test_paper_1
    python test_code_reviewer_0.py --code "import torch..." --paper-id test_paper_2
    python test_code_reviewer_0.py --paper-id test_paper_3  # Uses sample code
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import difflib

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
trn_execute_dir = os.path.join(project_root, 'trn_execute')
code_gen_dir = os.path.join(project_root, 'code_gen')
if trn_execute_dir not in sys.path:
    sys.path.insert(0, trn_execute_dir)
if code_gen_dir not in sys.path:
    sys.path.insert(0, code_gen_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from trn_execute (will need to mock S3 saving for local testing)
try:
    # Import the function directly
    from app import code_reviewer_0
    logger.info("✅ Successfully imported code_reviewer_0 from trn_execute/app.py")
except ImportError as e:
    logger.error(f"❌ Failed to import code_reviewer_0: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)

# Import S3 functions (don't mock - let it save normally)
try:
    from s3_code_storage import save_code, get_code
    logger.info("✅ S3 code storage available (will save to S3)")
except Exception as e:
    logger.warning(f"⚠️ Could not import S3 functions: {e}")

# Import OpenSearch client for fetching papers
try:
    from opensearch_client import OpenSearchClient
    OPENSEARCH_AVAILABLE = True
    logger.info("✅ OpenSearch client available")
except ImportError as e:
    OPENSEARCH_AVAILABLE = False
    OpenSearchClient = None
    logger.warning(f"⚠️ OpenSearch client not available: {e}")


def generate_code_for_paper(paper_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate code for a paper using ChunkedPyTorchGenerator (like the normal pipeline).
    
    Returns:
        Tuple of (generated_code, paper_summary) or (None, None) if generation failed
    """
    if not OPENSEARCH_AVAILABLE:
        logger.error("❌ OpenSearch not available - cannot fetch paper")
        return None, None
    
    try:
        # Import code generator
        try:
            from chunked_generator import ChunkedPyTorchGenerator
            logger.info("✅ Successfully imported ChunkedPyTorchGenerator")
        except ImportError as e:
            logger.error(f"❌ Failed to import ChunkedPyTorchGenerator: {e}")
            return None, None
        
        # Initialize generator
        logger.info("🔧 Initializing ChunkedPyTorchGenerator...")
        generator = ChunkedPyTorchGenerator(
            batch_size=8,
            max_pdf_chunks=15,
            pages_per_pdf_chunk=2
        )
        
        # Get paper from OpenSearch
        opensearch_client = generator.opensearch_client
        logger.info(f"🔍 Fetching paper {paper_id} from OpenSearch...")
        paper = opensearch_client.get_paper_by_id(paper_id)
        
        if not paper:
            logger.error(f"❌ Paper {paper_id} not found in OpenSearch")
            return None, None
        
        logger.info(f"✅ Found paper: {paper.get('title', 'Unknown Title')}")
        
        # Get paper summary
        paper_summary = opensearch_client.get_paper_summary(paper)
        
        # Generate code
        logger.info(f"🚀 Generating code for paper {paper_id}...")
        code_gen_result = generator.generate_code_for_paper(paper_id)
        
        if not code_gen_result.get("success") or not code_gen_result.get("code"):
            error = code_gen_result.get("error", "Code generation failed")
            logger.error(f"❌ Code generation failed: {error}")
            return None, paper_summary
        
        generated_code = code_gen_result["code"]
        logger.info(f"✅ Generated code (length: {len(generated_code)} chars)")
        
        return generated_code, paper_summary
        
    except Exception as e:
        logger.error(f"❌ Error generating code: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def get_sample_code() -> str:
    """Get sample code with TRN compatibility issues for testing"""
    return """import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Model definition
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Load dataset
train_dataset = MNIST(root='./data', train=True, download=True, transform=None)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(5):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
"""


def save_test_results(
    output_dir: Path,
    paper_id: str,
    original_code: str,
    reviewed_code: Optional[str],
    fixes_summary: list,
    code_changed: bool,
    bedrock_response: Optional[str] = None,
    error: Optional[str] = None
):
    """Save all test results to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original code (generated code before Code Reviewer 0)
    original_file = output_dir / f"{paper_id}_01_ORIGINAL_GENERATED.py"
    original_file.write_text(original_code, encoding='utf-8')
    logger.info(f"💾 Saved original generated code to {original_file}")
    
    # Save reviewed code (after Code Reviewer 0)
    if reviewed_code:
        reviewed_file = output_dir / f"{paper_id}_02_REVIEWED_BY_CODE_REVIEWER_0.py"
        reviewed_file.write_text(reviewed_code, encoding='utf-8')
        logger.info(f"💾 Saved reviewed code to {reviewed_file}")
        
        # Save diff if code changed
        if code_changed:
            diff_file = output_dir / f"{paper_id}_03_DIFF.txt"
            original_lines = original_code.splitlines(keepends=True)
            reviewed_lines = reviewed_code.splitlines(keepends=True)
            diff = difflib.unified_diff(
                original_lines,
                reviewed_lines,
                fromfile='01_ORIGINAL_GENERATED.py',
                tofile='02_REVIEWED_BY_CODE_REVIEWER_0.py',
                lineterm=''
            )
            diff_file.write_text(''.join(diff), encoding='utf-8')
            logger.info(f"💾 Saved diff to {diff_file}")
    else:
        logger.warning("⚠️ No reviewed code to save")
    
    # Save fixes summary
    fixes_file = output_dir / f"{paper_id}_04_FIXES_SUMMARY.txt"
    if fixes_summary:
        fixes_text = '\n'.join(f"- {fix}" if not fix.startswith('-') else fix for fix in fixes_summary)
    else:
        fixes_text = "No fixes applied" if not code_changed else "Fixes applied but summary not available"
    fixes_file.write_text(fixes_text, encoding='utf-8')
    logger.info(f"💾 Saved fixes summary to {fixes_file}")
    
    # Save bedrock response (raw)
    if bedrock_response:
        bedrock_file = output_dir / f"{paper_id}_05_BEDROCK_RESPONSE.txt"
        bedrock_file.write_text(bedrock_response, encoding='utf-8')
        logger.info(f"💾 Saved Bedrock response to {bedrock_file}")
    
    # Save test metadata
    metadata = {
        "paper_id": paper_id,
        "timestamp": datetime.now().isoformat(),
        "code_changed": code_changed,
        "original_code_length": len(original_code),
        "reviewed_code_length": len(reviewed_code) if reviewed_code else 0,
        "fixes_count": len(fixes_summary) if fixes_summary else 0,
        "fixes_summary": fixes_summary,
        "success": reviewed_code is not None,
        "error": error
    }
    
    metadata_file = output_dir / f"{paper_id}_00_TEST_METADATA.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info(f"💾 Saved test metadata to {metadata_file}")
    
    return metadata


def test_code_reviewer_0(
    code: str,
    paper_id: str,
    paper_summary: Optional[str] = None,
    output_dir: Path = Path("code_reviewer_0_testing")
) -> Dict[str, Any]:
    """Test Code Reviewer 0 with given code"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing Code Reviewer 0 for paper: {paper_id}")
    logger.info(f"{'='*80}\n")
    
    logger.info(f"Original code length: {len(code)} chars")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Call Code Reviewer 0
    try:
        result = code_reviewer_0(code, paper_id, paper_summary)
        
        if result:
            reviewed_code = result.get("code")
            fixes_summary = result.get("fixes_summary", [])
            code_changed = result.get("code_changed", False)
            
            logger.info(f"\n✅ Code Reviewer 0 completed successfully!")
            logger.info(f"   Code changed: {code_changed}")
            logger.info(f"   Original length: {len(code)} chars")
            logger.info(f"   Reviewed length: {len(reviewed_code)} chars" if reviewed_code else "   No reviewed code")
            logger.info(f"   Fixes applied: {len(fixes_summary)}")
            
            if fixes_summary:
                logger.info(f"\n   Fixes summary:")
                for fix in fixes_summary:
                    logger.info(f"     - {fix}")
            
            # Save results
            metadata = save_test_results(
                output_dir=output_dir,
                paper_id=paper_id,
                original_code=code,
                reviewed_code=reviewed_code,
                fixes_summary=fixes_summary,
                code_changed=code_changed
            )
            
            return {
                "success": True,
                "metadata": metadata,
                "result": result
            }
        else:
            error_msg = "Code Reviewer 0 returned None (failed to fix code)"
            logger.error(f"❌ {error_msg}")
            
            # Save results even on failure
            metadata = save_test_results(
                output_dir=output_dir,
                paper_id=paper_id,
                original_code=code,
                reviewed_code=None,
                fixes_summary=[],
                code_changed=False,
                error=error_msg
            )
            
            return {
                "success": False,
                "metadata": metadata,
                "error": error_msg
            }
            
    except Exception as e:
        error_msg = f"Error testing Code Reviewer 0: {str(e)}"
        logger.error(f"❌ {error_msg}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Save results even on exception
        metadata = save_test_results(
            output_dir=output_dir,
            paper_id=paper_id,
            original_code=code,
            reviewed_code=None,
            fixes_summary=[],
            code_changed=False,
            error=error_msg
        )
        
        return {
            "success": False,
            "metadata": metadata,
            "error": error_msg
        }


def main():
    parser = argparse.ArgumentParser(
        description='Test Code Reviewer 0 locally',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with paper ID (generates code first, then runs Code Reviewer 0)
  python test_code_reviewer_0.py --paper-id 5c4065371aa218edd77b5ec7f5379f447c916e5e11ba09d1c49e5fb86a53d968
  
  # Test with sample code
  python test_code_reviewer_0.py --use-sample --paper-id test_1
  
  # Test with a code file
  python test_code_reviewer_0.py --code-file ../pytorch/some_code.py --paper-id test_2
  
  # Test with inline code
  python test_code_reviewer_0.py --code "import torch..." --paper-id test_3
        """
    )
    parser.add_argument('--code-file', type=str, help='Path to Python code file to test')
    parser.add_argument('--code', type=str, help='Python code string to test')
    parser.add_argument('--paper-id', type=str, 
                       help='Paper ID for testing (if provided without --code-file/--code/--use-sample, fetches from OpenSearch/S3)')
    parser.add_argument('--paper-summary', type=str, help='Optional paper summary for context (overrides fetched summary)')
    parser.add_argument('--output-dir', type=str, default='code_reviewer_0_testing', 
                       help='Output directory for test results (default: code_reviewer_0_testing)')
    parser.add_argument('--use-sample', action='store_true', 
                       help='Use sample code with TRN compatibility issues')
    
    args = parser.parse_args()
    
    # Get code to test
    code = None
    paper_summary = args.paper_summary
    
    if args.code_file:
        code_file = Path(args.code_file)
        if not code_file.exists():
            logger.error(f"❌ Code file not found: {code_file}")
            sys.exit(1)
        code = code_file.read_text(encoding='utf-8')
        logger.info(f"📄 Loaded code from file: {code_file}")
    elif args.code:
        code = args.code
        logger.info(f"📝 Using code from command line argument")
    elif args.use_sample:
        code = get_sample_code()
        logger.info(f"📋 Using sample code with TRN compatibility issues")
    elif args.paper_id:
        # Generate code first (like the normal pipeline), then run Code Reviewer 0
        logger.info(f"🚀 Generating code for paper {args.paper_id} (like normal pipeline)...")
        code, fetched_summary = generate_code_for_paper(args.paper_id)
        if not code:
            logger.error(f"❌ Could not generate code for paper {args.paper_id}")
            logger.error("   Make sure the paper exists in OpenSearch")
            sys.exit(1)
        # Use fetched summary if not provided
        if not paper_summary:
            paper_summary = fetched_summary
        logger.info(f"✅ Generated code for {args.paper_id} - now running Code Reviewer 0...")
    else:
        logger.error("❌ Must provide --paper-id, --code-file, --code, or --use-sample")
        parser.print_help()
        sys.exit(1)
    
    # Default paper_id if not provided
    paper_id = args.paper_id or 'test_paper'
    
    # Test Code Reviewer 0
    output_dir = Path(args.output_dir)
    result = test_code_reviewer_0(
        code=code,
        paper_id=paper_id,
        paper_summary=paper_summary,
        output_dir=output_dir
    )
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Test Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Paper ID: {paper_id}")
    logger.info(f"Success: {result['success']}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    if result['success']:
        metadata = result['metadata']
        logger.info(f"Code changed: {metadata['code_changed']}")
        logger.info(f"Fixes applied: {metadata['fixes_count']}")
        logger.info(f"\n✅ Test completed successfully!")
        logger.info(f"   Check {output_dir.absolute()} for detailed results")
    else:
        logger.error(f"\n❌ Test failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()

