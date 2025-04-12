FROM python:3.10-slim

WORKDIR /app

# Install dependencies in a single layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first to leverage Docker caching
COPY requirements.txt .

# Install poetry and dependencies
RUN pip install -requirements.txt

# Copy the application code
COPY . .

# Create a non-root user to run the application
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app

USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the API port
EXPOSE 8000

# Run with gunicorn for better performance in production
# Using uvicorn workers - adjust worker count based on CPU cores (2-4 × num_cores + 1)
CMD ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]