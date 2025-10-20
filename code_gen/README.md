# Code Generation System

code_gen folder: retrieves research papers from OpenSearch and for each paper, generates PyTorch code using Claude via AWS Bedrock

## Features

- **OpenSearch Integration**: Retrieve papers from Open Search index
- **AWS Bedrock Integration**: Generate PyTorch code using Claude models
- **Multiple Search Methods**: Search by title, author, keywords, or get recent papers
- **Code Generation**: Generate complete PyTorch implementations with documentation
- **File Management**: Save generated code and metadata to files
- **CLI Support**: Command-line interface for easy usage
- **Lambda Support**: AWS Lambda handler for serverless deployment




## API Reference

### Actions

- `generate_by_id`: Generate code for a specific paper by ID
- `generate_by_ids`: Generate code for multiple papers by IDs
- `generate_by_title`: Generate code for papers matching a title
- `generate_by_author`: Generate code for papers by a specific author
- `generate_by_keywords`: Generate code for papers matching abstract keywords
- `generate_recent`: Generate code for recently ingested papers
- `get_paper_info`: Get paper information without generating code

### Parameters

- `paper_id`: OpenSearch document ID
- `paper_ids`: List of OpenSearch document IDs
- `title`: Paper title to search for
- `author`: Author name to search for
- `keywords`: Keywords to search in abstract
- `max_papers`: Maximum number of papers to process (default: 5)
- `include_full_content`: Whether to include full paper content (default: False)
- `days`: Days to look back for recent papers (default: 30)

## Examples

### Generate Code for a Specific Paper

```python
from code_gen.main_handler import CodeGenHandler

handler = CodeGenHandler()
result = handler.handle_lambda_event({
    'action': 'generate_by_id',
    'paper_id': 'paper_123',
    'include_full_content': True
})

if result['success']:
    print(f"Generated code for: {result['paper_title']}")
    print(f"Code length: {len(result['code'])}")
```

### Search and Generate Code

```python
# Generate code for papers about transformers
result = handler.handle_lambda_event({
    'action': 'generate_by_keywords',
    'keywords': 'transformer attention',
    'max_papers': 3,
    'include_full_content': False
})

for paper_result in result['results']:
    if paper_result['success']:
        print(f"Generated code for: {paper_result['paper_title']}")
```

### Save Generated Code

```python
# Generate and save code
result = handler.handle_lambda_event({
    'action': 'generate_by_title',
    'title': 'BERT',
    'max_papers': 1
})

if result['success'] and result['results']:
    paper_result = result['results'][0]
    if paper_result['success']:
        saved_path = handler.generator.save_generated_code(paper_result)
        print(f"Code saved to: {saved_path}")
```

## Testing

Run the test examples:

```bash
python -m code_gen.test_examples
```

This will test:
- OpenSearch connection
- Bedrock connection
- Code generation for sample papers
- File saving functionality

## AWS Lambda Deployment

The system is designed to work as an AWS Lambda function. Deploy using:

```bash
# Package the code
zip -r code_gen_lambda.zip code_gen/

# Deploy to AWS Lambda
aws lambda create-function \
  --function-name CodeGenerator \
  --runtime python3.9 \
  --role arn:aws:iam::your-account:role/lambda-execution-role \
  --handler code_gen.main_handler.lambda_handler \
  --zip-file fileb://code_gen_lambda.zip
```

## Error Handling

The system includes comprehensive error handling for:

- OpenSearch connection issues
- Bedrock API errors
- Invalid paper IDs
- Missing environment variables
- File I/O errors

All errors are logged and returned in a structured format.

## Configuration

Configuration is managed through environment variables and the `Config` class in `utils.py`. The system validates required configuration on startup.

## Dependencies

- `boto3`: AWS SDK
- `opensearch-py`: OpenSearch client
- `python-dotenv`: Environment variable management
- `requests`: HTTP client
- `urllib3`: HTTP library

## License

This code generation system is part of the Annapurna research paper processing pipeline.
