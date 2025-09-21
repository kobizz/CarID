FROM python:3.13-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service/*.py ./
COPY service/logging.yaml ./
COPY data/ ./data/

# Copy setup documentation
COPY ADDON_SETUP.md .

# Create necessary directories
RUN mkdir -p /app/data

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--log-config", "logging.yaml"]
