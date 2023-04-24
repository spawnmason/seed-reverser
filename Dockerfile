# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies
RUN apt-get update && apt-get install -y build-essential

# Build the CUDA executable
RUN nvcc -O3 -o seed-reverser main.cu

# Set the entry point to the CUDA executable
ENTRYPOINT ["/app/seed-reverser"]