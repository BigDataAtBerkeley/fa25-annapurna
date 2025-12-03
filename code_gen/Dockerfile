FROM public.ecr.aws/lambda/python:3.9

# Install system packages needed for PyMuPDF (build tools + libraries)
RUN yum update -y && \
    yum install -y gcc gcc-c++ make && \
    yum clean all

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install dependencies: try pre-built wheel first, fallback to source build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir boto3>=1.40.0 opensearch-py>=2.0.0 python-dotenv>=1.0.0 requests>=2.32.0 "urllib3>=1.26.0,<2.0.0" Pillow>=10.0.0 && \
    (pip install --no-cache-dir --only-binary=:all: pymupdf>=1.23.0 || \
     pip install --no-cache-dir pymupdf>=1.23.0)

# Copy function code
COPY *.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "lambda_handler.lambda_handler" ]

