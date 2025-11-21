FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY code-gen-for-deliv/requirements.txt /app/code-gen-for-deliv/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r code-gen-for-deliv/requirements.txt

# Copy application code
COPY . /app/

# Set Python path
ENV PYTHONPATH=/app:/app/code-gen-for-deliv:/app/trn-execute-for-deliv

# Create results directory
RUN mkdir -p /app/results

# Set default entrypoint
ENTRYPOINT ["python3", "pipeline_for_delivery.py"]

