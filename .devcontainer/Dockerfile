# Use NVIDIA PyTorch image as base
FROM nvcr.io/nvidia/pytorch:23.07-py3

# ENV DEBIAN_FRONTEND=noninteractive

# Install Conda
RUN apt update && apt install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/miniconda && \
    rm ~/miniconda.sh

# Set Conda in the PATH
ENV PATH="/opt/miniconda/bin:$PATH"

# Create a new Conda environment (Python 3.10)
RUN conda create -n myenv python=3.10 -y

# Set Conda environment activation
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate myenv" >> ~/.bashrc

# Install necessary packages inside Conda environment
RUN conda install -n myenv -y pip numpy opencv && \
    /opt/miniconda/bin/pip install torch_geometric

# Default to using Conda environment
ENV CONDA_DEFAULT_ENV=myenv
ENV PATH="/opt/miniconda/envs/myenv/bin:$PATH"

# Install FFmpeg and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install python package dependencies
RUN python -m pip install --upgrade pip
COPY requirements.txt /workspace/requirements.txt
RUN conda run -n myenv pip install --no-cache-dir -r requirements.txt
