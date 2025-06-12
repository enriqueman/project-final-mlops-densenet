FROM python:3.9

# Install required packages
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy requirements first for better caching
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install additional dependencies needed for model download
RUN pip install boto3

# Copy application code
COPY ./app /code/app
COPY ./scripts/download_model.py /code/download_model.py

# Create model directory
RUN mkdir -p /code/app/model

# Set environment variables
ARG AWS_REGION
ARG STAGE
ARG MODEL_REPOSITORY
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_ACCOUNT_ID

ENV AWS_DEFAULT_REGION=${AWS_REGION}
ENV STAGE=${STAGE}
ENV MODEL_REPOSITORY=${MODEL_REPOSITORY}
ENV MODEL_PATH=/code/app/model/densenet121_Opset17.onnx
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}

# Create startup script that downloads model and starts app
RUN echo '#!/bin/bash\n\
set -e\n\
echo "=== DESCARGANDO MODELO AL INICIO ==="\n\
python /code/download_model.py\n\
if [ ! -f "/code/app/model/densenet121_Opset17.onnx" ]; then\n\
    echo "Error: Modelo no descargado correctamente"\n\
    exit 1\n\
fi\n\
echo "=== INICIANDO APLICACIÓN ==="\n\
exec uvicorn app.app:app --host 0.0.0.0 --port 80' > /code/start.sh && chmod +x /code/start.sh

# Run the startup script
CMD ["/code/start.sh"]