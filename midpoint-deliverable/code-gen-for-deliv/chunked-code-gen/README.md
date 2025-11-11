# Chunked Code Generation

This module implements a chunked approach to generating PyTorch code from long research papers. Instead of trying to fit the entire paper into a single API call (which can exceed token limits), this approach:

1. **Splits the paper into 5 chunks**
2. **Generates detailed summaries for each chunk** (with mathematical formulas, key ideas, algorithms)
3. **Combines all summaries into a final code generation call**

## Why Use This?

- **Handles long papers**: Papers with 18M+ characters that would fail with standard approach
- **Better context**: Each chunk gets full attention for detailed summarization
- **Throttling mitigation**: Sequential processing with delays prevents rate limit errors
- **Cost effective**: Uses Claude 3 Haiku for chunk summaries (faster, cheaper)

## Usage

### Basic Usage

```python
from chunked_code_gen import ChunkedPyTorchGenerator

# Initialize generator
generator = ChunkedPyTorchGenerator(
    num_chunks=5,  # Split paper into 5 parts
    use_haiku_for_chunks=True  # Use Haiku for chunk summaries (faster, better rate limits)
)

# Generate code for a paper
result = generator.generate_code_for_paper(paper_id="C-hIZpoBclM7MZc3XpR7")

if result["success"]:
    print(f"Generated code: {len(result['code'])} characters")
    print(f"Processing time: {result['total_generation_time']:.1f}s")
    print(f"Successful chunks: {result['successful_chunks']}/{result['num_chunks']}")
else:
    print(f"Error: {result['error']}")
```

### Integration with Pipeline

You can integrate this into `pipeline_for_delivery.py` by replacing the standard generator:

```python
# Instead of:
from pytorch_generator import PyTorchCodeGenerator
generator = PyTorchCodeGenerator()

# Use:
from chunked_code_gen import ChunkedPyTorchGenerator
generator = ChunkedPyTorchGenerator(num_chunks=5, use_haiku_for_chunks=True)
```

## Architecture

### Components

1. **ChunkedBedrockClient** (`chunked_bedrock_client.py`)
   - Handles API calls to Bedrock
   - `summarize_chunk()`: Generates detailed summary for each chunk
   - `generate_final_code()`: Combines summaries into final code
   - Implements throttling mitigation (exponential backoff, delays)

2. **ChunkedPyTorchGenerator** (`chunked_generator.py`)
   - Orchestrates the chunked generation process
   - Splits papers into chunks
   - Processes chunks sequentially
   - Combines results

### Process Flow

```
Paper (18M chars)
    ↓
Split into 5 chunks (~3.6M chars each)
    ↓
Chunk 1 → Summarize (Haiku) → Detailed Summary
    ↓ (2s delay)
Chunk 2 → Summarize (Haiku) → Detailed Summary
    ↓ (2s delay)
Chunk 3 → Summarize (Haiku) → Detailed Summary
    ↓ (2s delay)
Chunk 4 → Summarize (Haiku) → Detailed Summary
    ↓ (2s delay)
Chunk 5 → Summarize (Haiku) → Detailed Summary
    ↓
Combine all summaries
    ↓
Generate final code (Sonnet) → Complete PyTorch Code
```

## Configuration

### Environment Variables

```bash
# Delay between chunk processing (seconds)
CHUNK_PROCESSING_DELAY=2.0

# Model IDs
BEDROCK_CHUNK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0  # For chunk summaries
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0        # For final code

# AWS Region
AWS_REGION=us-east-1
```

### Parameters

- `num_chunks`: Number of chunks to split paper into (default: 5)
- `use_haiku_for_chunks`: Use Claude 3 Haiku for chunk summaries (default: True)
  - **Pros**: Faster, better rate limits, cheaper
  - **Cons**: Slightly lower quality (but fine for summarization)

## Throttling Mitigation

See [THROTTLING_MITIGATION.md](THROTTLING_MITIGATION.md) for detailed strategies.

Key techniques:
1. **Sequential processing**: Process chunks one at a time
2. **Inter-chunk delays**: 2s delay between chunks (configurable)
3. **Exponential backoff**: Automatic retry with increasing delays
4. **Model selection**: Use Haiku for chunks (better rate limits)
5. **Jitter**: Random delays to avoid synchronized requests

## Output Format

The generator returns a dictionary with:

```python
{
    "success": True,
    "paper_id": "...",
    "paper_title": "...",
    "code": "...",  # Generated PyTorch code
    "model_used": "...",
    "dataset_recommendations": {...},
    "recommended_dataset": "imdb",
    "chunk_results": [
        {
            "chunk_number": 1,
            "success": True,
            "summary_length": 5000,
            "processing_time": 12.5
        },
        ...
    ],
    "num_chunks": 5,
    "successful_chunks": 5,
    "total_generation_time": 145.2,
    "final_generation_time": 45.8
}
```

## Performance

Expected performance for a typical long paper (18M chars):

- **Chunk processing**: ~5-15s per chunk × 5 = 25-75s
- **Inter-chunk delays**: 2s × 4 = 8s
- **Final generation**: ~30-60s
- **Total**: ~2-3 minutes

This is acceptable compared to:
- **Standard approach**: Fails immediately with "Input too long"
- **Chunked approach**: Successfully generates code

## Error Handling

- If a chunk fails, processing continues with other chunks
- Need at least 50% of chunks to succeed (configurable)
- Final generation will use whatever summaries are available
- Detailed error information in `chunk_results`

## Comparison with Standard Approach

| Feature | Standard | Chunked |
|---------|----------|---------|
| Max paper size | ~150k chars | Unlimited |
| API calls | 1 | 6 (5 chunks + 1 final) |
| Processing time | ~30s | ~2-3 min |
| Throttling risk | Low | Mitigated |
| Cost | Lower | Higher (6 calls) |
| Success rate | Fails on long papers | Works on all papers |

## Future Improvements

1. **Parallel chunk processing**: Process chunks in parallel (with rate limiting)
2. **Smart chunking**: Split at section boundaries instead of equal sizes
3. **Caching**: Cache chunk summaries for papers
4. **Adaptive chunking**: Adjust number of chunks based on paper length
5. **Progress tracking**: Real-time progress updates

