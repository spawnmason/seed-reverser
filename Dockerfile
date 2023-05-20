# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

# Set the working directory to /app
WORKDIR /data

# Install any necessary dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy the current directory contents into the container at /app
COPY . /app

# Build the CUDA executable
RUN nvcc -O3 -o seed-reverser main.cu

# Set the entry point to the CUDA executable
ENTRYPOINT ["/app/seed-reverser"]
