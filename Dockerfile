# Use the Python 3.11 image
FROM python:3.11

# Set the working directory in the container
WORKDIR /greenroom_subjects

# Copy requirements.txt to the container and install dependencies
COPY ../requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all files from the current directory to the container
COPY . .

# Expose the Flask port
EXPOSE 5005

# Run the Flask application
CMD ["python", "app/app.py"]