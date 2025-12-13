FROM public.ecr.aws/lambda/python:3.10
RUN yum update -y && \
    yum install -y gcc gcc-c++ make && \
    yum clean all
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir --only-binary=:all: pymupdf>=1.23.0 || \
    (echo "❌ ERROR: Failed to install pymupdf from pre-built wheels." && \
     echo "   Cannot build from source - Lambda base image g++ doesn't support C++20." && \
     echo "   Try: pip install --only-binary=:all: pymupdf==1.23.0 (specific version)" && \
     exit 1)

RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt
COPY *.py ${LAMBDA_TASK_ROOT}/
COPY page_classifier_model.pkl ${LAMBDA_TASK_ROOT}/
CMD [ "lambda_handler.lambda_handler" ]

