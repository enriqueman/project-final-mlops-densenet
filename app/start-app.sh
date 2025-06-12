#!/bin/bash
set -e

echo "=== INICIANDO APLICACIÓN DENSENET ==="

# Variables de entorno
AWS_REGION=${AWS_REGION:-us-east-1}
STAGE=${STAGE:-dev}
MODEL_REPOSITORY=${MODEL_REPOSITORY:-densenet121-model-dev}
ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}

echo "Configuración:"
echo "  AWS_REGION: $AWS_REGION"
echo "  STAGE: $STAGE"
echo "  MODEL_REPOSITORY: $MODEL_REPOSITORY"
echo "  ACCOUNT_ID: $ACCOUNT_ID"

# Verificar si el modelo ya existe
if [ ! -f "/code/app/model/densenet121_Opset17.onnx" ]; then
    echo "=== DESCARGANDO MODELO DESDE ECR ==="
    python /code/download_model_runtime.py
    
    # Verificar que el modelo se descargó correctamente
    if [ ! -f "/code/app/model/densenet121_Opset17.onnx" ]; then
        echo "❌ Error: No se pudo descargar el modelo"
        exit 1
    fi
    
    MODEL_SIZE=$(du -h /code/app/model/densenet121_Opset17.onnx | cut -f1)
    echo "✅ Modelo descargado correctamente: $MODEL_SIZE"
else
    MODEL_SIZE=$(du -h /code/app/model/densenet121_Opset17.onnx | cut -f1)
    echo "Modelo ya existe localmente: $MODEL_SIZE"
fi

# Iniciar aplicación
echo "=== INICIANDO FASTAPI ==="
echo "Iniciando servidor en puerto 80..."

exec uvicorn app.app:app \
    --host 0.0.0.0 \
    --port 80 \
    --workers 1 \
    --log-level info \
    --access-log