# Dockerfile for PIM Matmul Benchmarks Development and Testing
# Ubuntu 20.04 with UPMEM SDK, build tools, and Python environment

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential python3 python3-pip wget sudo git pkg-config && \
    apt-get clean

# Install UPMEM SDK
RUN wget http://sdk-releases.upmem.com/2025.1.0/debian_10/upmem-2025.1.0-Linux-x86_64.tar.gz && \
    tar -xzf upmem-2025.1.0-Linux-x86_64.tar.gz -C /opt && \
    rm upmem-2025.1.0-Linux-x86_64.tar.gz

# Install Python dependencies
RUN pip3 install --upgrade pip && pip3 install pyyaml

# Set environment variables for UPMEM SDK
ENV PKG_CONFIG_PATH="/opt/upmem-2025.1.0-Linux-x86_64/share/pkgconfig"
ENV PATH="/opt/upmem-2025.1.0-Linux-x86_64/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/upmem-2025.1.0-Linux-x86_64/lib"

# Set up workdir and copy project
WORKDIR /workspace
COPY . /workspace

SHELL ["/bin/bash", "-c"]

# Source the upmem_env.sh script
RUN source /opt/upmem-2025.1.0-Linux-x86_64/upmem_env.sh simulator

# Source the environment script
RUN source /workspace/source.me

