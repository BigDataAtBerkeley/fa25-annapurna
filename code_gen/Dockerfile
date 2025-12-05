FROM public.ecr.aws/lambda/python:3.9

# Install system packages needed for PyMuPDF (build tools + libraries)
RUN yum update -y && \
    yum install -y gcc gcc-c++ make && \
    yum clean all

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install dependencies: try pre-built wheel first, fallback to source build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt && \
    (pip install --no-cache-dir --only-binary=:all: pymupdf>=1.23.0 || \
     pip install --no-cache-dir pymupdf>=1.23.0)

# Copy function code
COPY *.py ${LAMBDA_TASK_ROOT}/

# Copy classifier model file (required for page relevance classification)
COPY page_classifier_model.pkl ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "lambda_handler.lambda_handler" ]

