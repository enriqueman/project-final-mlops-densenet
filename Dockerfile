FROM python:3.9

# Install required packages for model download
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy requirements first for better caching
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install additional dependencies needed for model download
RUN pip install docker boto3

# Copy application code
COPY ./app /code/app
COPY ./scripts/download_model.py /code/download_model.py

# Create model directory
RUN mkdir -p /code/app/model

# Set environment variables
ARG AWS_REGION
ARG STAGE
ARG MODEL_REPOSITORY
ENV AWS_DEFAULT_REGION=${AWS_REGION}
ENV STAGE=${STAGE}
ENV MODEL_REPOSITORY=${MODEL_REPOSITORY}
ENV MODEL_PATH=/code/app/model/densenet121_Opset17.onnx

# Download model during build
RUN python /code/download_model.py

# Run the FastAPI application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]