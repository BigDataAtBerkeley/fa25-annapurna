# Code Generation Testing Utilities

This is to test code gen working WITHOUT TOUCHING AWS (so no save to S3, OpenSearcgh, etc). It just pulls papers from S3,
generates their code, and then saves the generated py files to your working directory called "generated_code/" 

## To run this just run: test_code_generation.py


### To find which paper IDs you can either find their IDs using `python check_opensearch.py` OR you can use the file
`random_sample_generate.py`, which willpull 5 random papers from s3 and generate their code
   ```

#### Sample 5 random papers (default):

```bash
python code-gen-testing/random_sample_generate.py
```

#### Sample 10 random papers:

```bash
python code-gen-testing/random_sample_generate.py --count 10
```

### What it does:

1. **Randomly samples** papers from OpenSearch (default: 5 papers)
2. **Generates code** for each paper using the test code generator
3. **Saves all results** to `generated_code/` directory (clears it first by default)
4. **Prints summary** of successful and failed generations

## test_code_generation.py Details

### What it does:

1. **Downloads paper from S3** (or fetches metadata from OpenSearch if using `--paper-id`)
2. **Analyzes paper** to recommend appropriate datasets
3. **Generates PyTorch code** using AWS Bedrock (Claude)
4. **Saves code locally** to your working directory (or specified output directory)
5. **Saves metadata** as JSON alongside the code file

### Output:

For each successful generation, you'll get:
- `{paper_title}_{timestamp}.py` - The generated PyTorch code
- `{paper_title}_{timestamp}_metadata.json` - Metadata about the paper and generation

