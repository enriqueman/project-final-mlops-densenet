# Dockerfile optimizado para EC2
FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Crear directorio para el modelo
RUN mkdir -p /app/model

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY app.py .
COPY start-app.sh .
RUN chmod +x start-app.sh

# Configurar logging
RUN mkdir -p /var/log/app
VOLUME ["/var/log/app"]

# Exponer puertos
EXPOSE 8000

# Variables de entorno
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model/densenet121_Opset17.onnx

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Comando para ejecutar la aplicación
CMD ["./start-app.sh"]