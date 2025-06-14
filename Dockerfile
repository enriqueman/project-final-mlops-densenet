# Dockerfile modificado para obtener modelo desde S3
FROM python:3.9-slim

# Argumentos de construcción
ARG AWS_REGION=us-east-1
ARG STAGE=dev
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_ACCOUNT_ID

# Variables de entorno
ENV AWS_REGION=${AWS_REGION}
ENV STAGE=${STAGE}
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws/

# Crear directorio de trabajo
WORKDIR /code

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Crear directorio para el modelo
RUN mkdir -p /code/app/model

# Copiar código de la aplicación
COPY app/ ./app/
COPY scripts/download_model_from_s3.py ./scripts/

# Hacer ejecutable el script de descarga
RUN chmod +x ./scripts/download_model_from_s3.py

# Script de inicialización que descarga el modelo y luego inicia la app
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Exponer puerto
EXPOSE 80

# Usar el script de entrada
ENTRYPOINT ["./docker-entrypoint.sh"]