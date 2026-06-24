# Use official Python slim image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (layer caching — faster rebuilds)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app
ENV PYTHONPATH=/app/src
# Expose the port the app runs on
EXPOSE 8000


# Start FastAPI with uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]