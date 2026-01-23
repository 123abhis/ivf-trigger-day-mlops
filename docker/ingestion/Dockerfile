FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (layer caching)
COPY docker/ingestion/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy ingestion code
COPY docker/ingestion/store_to_clickhouse.py .

# Default command
CMD ["python", "store_to_clickhouse.py"]

ENV GIT_PYTHON_REFRESH=quiet
