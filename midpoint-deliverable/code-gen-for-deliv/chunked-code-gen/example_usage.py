#!/usr/bin/env python3
"""
Example usage of the chunked code generator.
"""

import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunked_code_gen import ChunkedPyTorchGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Example usage of chunked code generator."""
    
    # Initialize generator
    print("Initializing Chunked PyTorch Code Generator...")
    generator = ChunkedPyTorchGenerator(
        num_chunks=5,
        use_haiku_for_chunks=True  # Use Haiku for chunk summaries (faster, better rate limits)
    )
    
    # Example paper ID (replace with actual paper ID)
    paper_id = "C-hIZpoBclM7MZc3XpR7"
    
    print(f"\nGenerating code for paper: {paper_id}")
    print("=" * 80)
    
    # Generate code
    result = generator.generate_code_for_paper(paper_id)
    
    # Display results
    print("\n" + "=" * 80)
    if result["success"]:
        print("✅ Code generation successful!")
        print(f"\nPaper: {result.get('paper_title', 'Unknown')}")
        print(f"Code length: {len(result['code']):,} characters")
        print(f"Total time: {result['total_generation_time']:.1f}s")
        print(f"Successful chunks: {result['successful_chunks']}/{result['num_chunks']}")
        print(f"Recommended dataset: {result.get('recommended_dataset', 'N/A')}")
        
        # Show chunk results
        print("\nChunk Results:")
        for chunk_result in result.get('chunk_results', []):
            status = "✅" if chunk_result.get('success') else "❌"
            chunk_num = chunk_result.get('chunk_number', '?')
            time_taken = chunk_result.get('processing_time', 0)
            print(f"  {status} Chunk {chunk_num}: {time_taken:.1f}s")
            if not chunk_result.get('success'):
                print(f"     Error: {chunk_result.get('error', 'Unknown')}")
        
        # Save code to file
        output_file = f"generated_code_{paper_id}.py"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['code'])
        print(f"\n✅ Code saved to: {output_file}")
        
    else:
        print("❌ Code generation failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
        
        # Show chunk results if available
        if 'chunk_results' in result:
            print("\nChunk Results:")
            for chunk_result in result['chunk_results']:
                status = "✅" if chunk_result.get('success') else "❌"
                chunk_num = chunk_result.get('chunk_number', '?')
                print(f"  {status} Chunk {chunk_num}")
                if not chunk_result.get('success'):
                    print(f"     Error: {chunk_result.get('error', 'Unknown')}")

if __name__ == "__main__":
    main()

