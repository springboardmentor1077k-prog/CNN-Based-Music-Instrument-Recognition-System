# Use NVIDIA optimized TensorFlow image for DGX A100
FROM nvcr.io/nvidia/tensorflow:22.11-tf2-py3

# Prevent Python from writing .pyc files (avoids permission issues)
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory (standard for this base image)
WORKDIR /workspace

# Install system dependencies required for Audio Processing (Librosa/Soundfile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Ensure output directories exist and are writable by any user (for non-root execution)
RUN mkdir -p logs outputs && chmod -R 777 logs outputs

# Set the default command to show help
CMD ["python", "src/training.py", "--help"]
