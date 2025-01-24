# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
  python3.8 \
  python3-pip \
  libsndfile1 \
  ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/input data/output models/checkpoints

# Create entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
