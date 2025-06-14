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
    MODEL_PATH="/code/app/model/densenet121_Opset17.onnx"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "❌ El archivo del modelo no existe después de la descarga: $MODEL_PATH"
        echo "📁 Contenido del directorio model:"
        ls -la /code/app/model/ || echo "Directorio no existe"
        exit 1
    fi
    
    # Verificar tamaño del modelo
    MODEL_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null || echo "0")
    echo "✅ Modelo verificado correctamente - Tamaño: $(echo $MODEL_SIZE | awk '{print $1/1024/1024 " MB"}')"
    
    # Establecer variable de entorno para FastAPI
    export MODEL_PATH="$MODEL_PATH"
}

# Función para verificar salud de la aplicación
health_check() {
    echo "🏥 Iniciando verificación de salud..."
    # Verificar que Python puede importar las dependencias principales
    python3 -c "import onnxruntime; import fastapi; import numpy; print('✅ Dependencias principales OK')" || {
        echo "❌ Error en dependencias de Python"
        exit 1
    }
    echo "✅ Verificación de salud completada"
}

# Función principal
main() {
    echo "🎯 Ejecutando script de entrada para FastAPI + DenseNet121"
    
    # Descargar modelo
    download_model
    
    # Verificar salud
    health_check
    
    echo "🚀 Iniciando aplicación FastAPI..."
    echo "📊 Variables de entorno relevantes:"
    echo "   STAGE: $STAGE"
    echo "   AWS_REGION: $AWS_REGION"
    echo "   MODEL_PATH: $MODEL_PATH"
    
    # Cambiar al directorio de la aplicación
    cd /code/app
    
    # Iniciar la aplicación con uvicorn
    exec uvicorn main:app --host 0.0.0.0 --port 80 --workers 1 --log-level info
}

# Ejecutar función principal
main "$@"