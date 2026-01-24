# Use a CUDA-enabled base image with Python
# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# PyTorch 2.2.0+cu121 with CUDA 1201 and Python 3.10.13 

# Set the working directory
WORKDIR /app

# Install system dependencies, build tools for C++ components, and Python
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    cmake \
    g++ \
    build-essential \
    libopencv-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

# Update to use python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Copy the requirements file and install Python dependencies
COPY requirements.txt .


# Copy the entire project
COPY . .

# Install CUDA-enabled PyTorch 2.2.0 (cu121)
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install CUDA-enabled xformers
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu121

RUN pip install -e .
# Install gsplat (CUDA extension)
# RUN pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

# Install project extras
# RUN pip install -e ".[all]"

# RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
# RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# RUN pip install xformers torch\>=2 torchvision

# Verified to work with CUDA 11.8 and torch 2.2.0
# RUN pip install --no-cache-dir -r requirements.txt


# Build the C++ component (DPRetrieval)
# WORKDIR /app/DPRetrieval
# RUN mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Return to the main working directory
WORKDIR /app

# Define a default command (this might need to be adjusted based on the actual entry point of the application)
# For now, it will just keep the container running.
# CMD ["python3", "vggt_long.py", "--image_dir", "./images/icar_test_1"]

# to build: `docker build -t vggt-long .`