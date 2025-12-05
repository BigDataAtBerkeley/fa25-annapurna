FROM public.ecr.aws/lambda/python:3.10

# Install system packages if needed for other dependencies
RUN yum update -y && \
    yum install -y gcc gcc-c++ make && \
    yum clean all

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install dependencies: Use pre-built wheels for pymupdf (avoids C++20 compilation issues)
# Container-based Lambdas don't support layers, so we must install pymupdf in the image
# The Lambda base image's g++ doesn't support C++20, so we must use pre-built wheels
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install pymupdf FIRST using pre-built wheel (avoids C++20 compilation issues)
# CRITICAL: --only-binary=:all: prevents building from source
RUN pip install --no-cache-dir --only-binary=:all: pymupdf>=1.23.0 || \
    (echo "❌ ERROR: Failed to install pymupdf from pre-built wheels." && \
     echo "   Cannot build from source - Lambda base image g++ doesn't support C++20." && \
     echo "   Try: pip install --only-binary=:all: pymupdf==1.23.0 (specific version)" && \
     exit 1)

# Install other dependencies (pymupdf already satisfied, pip will skip reinstalling it)
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy function code
COPY *.py ${LAMBDA_TASK_ROOT}/

# Copy classifier model file (required for page relevance classification)
COPY page_classifier_model.pkl ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "lambda_handler.lambda_handler" ]

