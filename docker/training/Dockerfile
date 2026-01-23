FROM python:3.10-slim

WORKDIR /app

COPY docker/training/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 👇 USE YOUR EXISTING TRAIN FILE (NO DUPLICATE LOGIC)
COPY src/train_model.py ./train_model.py

CMD ["python", "train_model.py"]

ENV GIT_PYTHON_REFRESH=quiet

