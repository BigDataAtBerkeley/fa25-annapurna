# Dataset Matching Strategy

## Simple Approach: Claude Selects During Code Generation

**Solution:** Claude automatically selects the appropriate dataset during code generation based on the paper's content.

### How It Works

1. **Prompt instructs Claude** to analyze the paper and choose the right dataset
2. **Claude reads paper abstract/content** to understand the domain
3. **Claude generates code** with the appropriate `load_dataset()` call
4. **Code runs on Trainium** with the pre-selected dataset

### Selection Logic (Built into Prompt)

```
Paper Type                      →  Dataset Choice
────────────────────────────────────────────────────
Vision / CNN / Image            →  'cifar10' or 'mnist'
NLP / Text / Transformers       →  'imdb' or 'wikitext2'
Language Models                 →  'wikitext2'
Generic / Simple Models         →  'synthetic'
```

### Example Generated Code

**For a CNN paper:**
```python
from dataset_loader import load_dataset

print("Using dataset: cifar10 (matches paper's vision task)")
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# Rest of model code...
```

**For an NLP paper:**
```python
from dataset_loader import load_dataset

print("Using dataset: imdb (matches paper's text classification task)")
train_data, test_data = load_dataset('imdb')

# Rest of model code...
```

### Why This Works

✅ **No extra API calls** - Claude already reads the paper for code generation
✅ **Simple** - One prompt change, no additional infrastructure
✅ **Accurate** - Claude understands paper content and can match appropriately
✅ **Explicit** - Generated code prints which dataset it's using and why
✅ **Cached** - Datasets are static on Trainium (one-time download from S3)

### Prompt Changes Made

Updated `code_gen/bedrock_client.py` prompt to:
1. **Require** `from dataset_loader import load_dataset` at the top
2. **List** all available datasets with descriptions
3. **Instruct** Claude to select based on paper domain
4. **Forbid** using torchvision.datasets or generating synthetic data
5. **Require** explanation of why dataset was chosen

### Verification

When code is generated, check:
```python
# Vision paper should have:
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# NLP paper should have:
train_data, test_data = load_dataset('imdb')
```

### Fallback Strategy

If Claude somehow picks the wrong dataset:
- Code will still run (all datasets work with PyTorch)
- Metrics will show which dataset was actually used
- Can manually test with different dataset using `test_code_on_trainium.py`

### Testing

```bash
# Generate code for a vision paper
aws lambda invoke \
  --function-name PapersCodeGenerator \
  --payload '{"action":"generate_by_id","paper_id":"<VISION_PAPER_ID>"}' \
  response.json

# Check generated code uses cifar10
cat response.json | jq -r '.code' | grep 'load_dataset'

# Generate code for NLP paper
aws lambda invoke \
  --function-name PapersCodeGenerator \
  --payload '{"action":"generate_by_id","paper_id":"<NLP_PAPER_ID>"}' \
  response.json

# Check generated code uses imdb
cat response.json | jq -r '.code' | grep 'load_dataset'
```

### Monitoring

After testing, OpenSearch will include:
```json
{
  "paper_id": "abc123",
  "dataset_name": "cifar10",  // Extracted from code output
  "test_success": true,
  "execution_time": 245.3
}
```

## Alternative Approaches (Not Implemented)

### ❌ Runtime Detection
- Code analyzes itself to pick dataset
- Too complex, runtime overhead

### ❌ Separate Claude Call
- Extra API call to match dataset
- Unnecessary cost and latency

### ❌ Manual Mapping
- Maintain a paper_type → dataset mapping
- Requires classification step

## Summary

**We use the simplest approach:** Claude selects the dataset during code generation based on paper content. This requires **zero additional API calls** and leverages Claude's existing understanding of the paper.

The prompt now explicitly instructs Claude to:
1. Import `dataset_loader`
2. Choose appropriate dataset based on paper domain
3. Use that dataset in the generated code
4. Explain the choice

✅ **Implementation complete** - No additional code needed!

