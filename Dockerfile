FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies for torchvision/transformers/PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (lightweight, no CUDA)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN sed -i '/^--extra-index-url.*cu/d' requirements.txt && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Default command (overridden by docker-compose per service)
CMD ["uvicorn", "be.main:app", "--host", "0.0.0.0", "--port", "8000"]
