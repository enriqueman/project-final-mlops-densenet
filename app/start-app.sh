#!/bin/bash
set -e

echo "=== INICIANDO APLICACIÓN DENSENET EN EC2 ==="

# Variables de entorno
AWS_REGION=${AWS_REGION:-us-east-1}
STAGE=${STAGE:-dev}
MODEL_ECR_REPOSITORY=${MODEL_ECR_REPOSITORY:-densenet121-model-dev}
ACCOUNT_ID=${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}

echo "Configuración:"
echo "  AWS_REGION: $AWS_REGION"
echo "  STAGE: $STAGE"
echo "  MODEL_REPOSITORY: $MODEL_ECR_REPOSITORY"
echo "  ACCOUNT_ID: $ACCOUNT_ID"

# Función para descargar modelo desde ECR
download_model() {
    echo "=== DESCARGANDO MODELO DESDE ECR ==="
    
    MODEL_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$MODEL_ECR_REPOSITORY:latest"
    echo "URI del modelo: $MODEL_URI"
    
    # Login a ECR
    echo "Autenticando con ECR..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    
    # Crear contenedor temporal para extraer modelo
    echo "Creando contenedor temporal para extraer modelo..."
    CONTAINER_ID=$(docker create $MODEL_URI)
    
    # Extraer modelo
    echo "Extrayendo modelo..."
    docker cp $CONTAINER_ID:/opt/ml/model/densenet121_Opset17.onnx /app/model/densenet121_Opset17.onnx
    
    # Limpiar contenedor temporal
    docker rm $CONTAINER_ID
    
    # Verificar que el modelo se descargó correctamente
    if [ -f "/app/model/densenet121_Opset17.onnx" ]; then
        MODEL_SIZE=$(du -h /app/model/densenet121_Opset17.onnx | cut -f1)
        echo "✅ Modelo descargado correctamente: $MODEL_SIZE"
    else
        echo "❌ Error: No se pudo descargar el modelo"
        exit 1
    fi
}

# Verificar si el modelo ya existe
if [ ! -f "/app/model/densenet121_Opset17.onnx" ]; then
    echo "Modelo no encontrado, descargando desde ECR..."
    download_model
else
    MODEL_SIZE=$(du -h /app/model/densenet121_Opset17.onnx | cut -f1)
    echo "Modelo ya existe localmente: $MODEL_SIZE"
fi

# Configurar logging
echo "=== CONFIGURANDO LOGS ==="
mkdir -p /var/log/app
touch /var/log/app/densenet.log
chmod 666 /var/log/app/densenet.log

# Iniciar aplicación
echo "=== INICIANDO FASTAPI ==="
echo "Iniciando servidor en puerto 8000..."

exec uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log \
    --log-config-disable-existing-loggers false