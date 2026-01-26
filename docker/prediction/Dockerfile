FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY docker/training/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src /app/src

ENV PYTHONPATH=/app

CMD ["python", "src/prediction.py"]
