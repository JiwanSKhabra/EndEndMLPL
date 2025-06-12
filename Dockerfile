# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file
COPY api/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn

# Copy the API code
COPY api/ ./api/

# Copy model directory (for when you have a model)
COPY model/ ./model/

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]