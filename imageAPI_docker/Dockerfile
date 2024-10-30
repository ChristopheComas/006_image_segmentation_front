# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file to install dependencies first (for caching purposes)
COPY imageAPI_docker/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY imageAPI_docker/ .

# Command to run your application
CMD ["python", "main.py"]
