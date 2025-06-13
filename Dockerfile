# Use Python 3.12-slim as the base image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy needed folders
COPY api/ ./api/
COPY model/ ./model/
COPY data/scaler.pkl ./data/scaler.pkl
COPY data/price_scaler.pkl ./data/price_scaler.pkl

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]