ARG CUDA_IMAGE="12.1.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

WORKDIR /app

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

# Install python and necessary packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install packages
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy all files to the image
COPY . .

# Expose the Flask port
EXPOSE 5005

# Run the Flask application
CMD ["python3", "app/app.py"]