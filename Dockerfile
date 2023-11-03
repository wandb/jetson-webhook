# Start from the official PyTorch image to avoid installing PyTorch manually
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Set the working directory to /app
WORKDIR /webhook

# Copy the current directory contents into the container at /app
COPY . /webhook

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the secrets.json to the container (Consider using Docker secrets or environment variables in production)
COPY secrets.json /app/webhook/secrets.json

# Expose the port the server is running on
EXPOSE 8090

# Run webhook.py when the container launches
CMD ["python", "webhook/webhook.py"]
