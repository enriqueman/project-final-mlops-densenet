#!/bin/bash

# Script de entrada para Docker - descarga modelo desde S3 e inicia la aplicación

set -e

echo "🚀 Iniciando contenedor para stage: $STAGE"
echo "📍 Región AWS: $AWS_REGION"

# Función para descargar modelo desde S3
download_model() {
    echo "📥 Descargando modelo desde S3..."
    
    # Configurar credenciales AWS si están disponibles
    if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
        echo "🔑 Configurando credenciales AWS..."
        aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
        aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
        aws configure set default.region "$AWS_REGION"
    fi
    
    # Intentar descargar el modelo usando el script de Python
    if python3 scripts/download_model_from_s3.py; then
        echo "✅ Modelo descargado exitosamente desde S3"
    else
        echo "❌ Error descargando modelo desde S3"
        exit 1
    fi
    
    # Verificar que el modelo existe
    if [ ! -f "/code/app/model/densenet121_Opset17.onnx" ]; then
        echo "❌ El archivo del modelo no existe después de la descarga"
        exit 1
    fi
    
    echo "✅ Modelo verificado correctamente"
}

# Función para verificar salud de la aplicación
health_check() {
    echo "🏥 Iniciando verificación de salud..."
    # Aquí puedes agregar verificaciones adicionales si es necesario
    return 0
}

# Función principal
main() {
    echo "🎯 Ejecutando script de entrada para FastAPI + DenseNet121"
    
    # Descargar modelo
    download_model
    
    # Verificar salud
    health_check
    
    echo "🚀 Iniciando aplicación FastAPI..."
    
    # Cambiar al directorio de la aplicación
    cd /code/app
    
    # Iniciar la aplicación con uvicorn
    exec uvicorn main:app --host 0.0.0.0 --port 80 --workers 1
}

# Ejecutar función principal
main "$@"