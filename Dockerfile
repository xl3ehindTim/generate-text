FROM nvidia/cuda:11.6.0-cudnn8.2.0-devel  # Base image with CUDA and cuDNN

# Set the working directory in the container
WORKDIR /greenroom_tagging

# Copy requirements.txt to the container and install dependencies
# remove ../?
COPY ../requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install gunicorn

# Copy all files from the current directory to the container
COPY . .

# Expose the Flask port
EXPOSE 5004

# Run the Flask application
CMD ["python", "app/app.py"]