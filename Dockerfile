# Dockerfile for PIM Matmul Benchmarks Development and Testing
# Ubuntu 20.04 with UPMEM SDK, build tools, and Python environment

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential python3 python3-pip wget sudo && \
    apt-get clean

# Install UPMEM SDK
RUN wget http://sdk-releases.upmem.com/2025.1.0/ubuntu_20.04/upmem_2025.1.0_amd64.deb && \
    dpkg -i upmem_2025.1.0_amd64.deb || apt-get install -f -y

# Install Python dependencies
RUN pip3 install --upgrade pip && pip3 install pyyaml

# Set up workdir and copy project
WORKDIR /workspace
COPY . /workspace

# Default command
CMD ["bash"]
