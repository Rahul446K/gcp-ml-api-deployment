# 1) Base image: Python 3.10, small Debian-based
FROM python:3.10-slim

# 2) Set working directory inside the container
WORKDIR /app

# 3) Install system deps (for transformers, tokenizers, etc.)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4) Copy Python dependencies file (we'll use inline pip for simplicity)
# (we skip requirements.txt and just pip install directly)

# 5) Install Python libraries
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    transformers \
    peft \
    accelerate \
    sentencepiece \
    torch

# 6) Copy your LoRA adapter and server code into the image
COPY lora-devotee/ /app/lora-devotee/
COPY server.py /app/server.py

# 7) Expose port 8000 inside the container
EXPOSE 8000

# 8) Command to run the API server when container starts
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
