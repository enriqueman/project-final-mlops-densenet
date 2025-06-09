#!/bin/bash

# Configuración
STAGE=${1:-dev}
AWS_REGION=${2:-us-east-1}
MODEL_FILE="models/densenet121_Opset17.onnx"

# Verificar que el modelo existe
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: El archivo $MODEL_FILE no existe"
    exit 1
fi

echo "🚀 Configurando repositorio ECR para modelos..."

# Obtener Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
MODEL_REPO_NAME="densenet121-model-$STAGE"
MODEL_ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$MODEL_REPO_NAME"

# Crear repositorio si no existe
echo "📦 Creando repositorio ECR: $MODEL_REPO_NAME"
aws ecr describe-repositories --repository-names $MODEL_REPO_NAME --region $AWS_REGION 2>/dev/null || {
    echo "Creando nuevo repositorio..."
    aws ecr create-repository \
        --repository-name $MODEL_REPO_NAME \
        --region $AWS_REGION \
        --image-scanning-configuration scanOnPush=true \
        --image-tag-mutability MUTABLE
}

# Configurar lifecycle policy
echo "⚙️  Configurando política de retención..."
aws ecr put-lifecycle-policy \
    --repository-name $MODEL_REPO_NAME \
    --region $AWS_REGION \
    --lifecycle-policy-text '{
        "rules": [
            {
                "rulePriority": 1,
                "description": "Keep last 3 model versions",
                "selection": {
                    "tagStatus": "any",
                    "countType": "imageCountMoreThan",
                    "countNumber": 3
                },
                "action": {
                    "type": "expire"
                }
            }
        ]
    }'

echo "🔨 Construyendo imagen del modelo..."

# Crear directorio temporal para la imagen
mkdir -p temp-model-build
cp "$MODEL_FILE" temp-model-build/densenet121_Opset17.onnx

# Crear Dockerfile para el modelo
cat > temp-model-build/Dockerfile << EOF
FROM scratch
COPY densenet121_Opset17.onnx /model/densenet121_Opset17.onnx
LABEL model.name="densenet121"
LABEL model.version="1.0"
LABEL model.format="onnx"
LABEL build.timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF

# Construir imagen
cd temp-model-build
docker build -t densenet121-model:latest .
cd ..

# Login a ECR
echo "🔐 Autenticando con ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $MODEL_ECR_URI

# Tag y push
echo "📤 Subiendo modelo a ECR..."
docker tag densenet121-model:latest $MODEL_ECR_URI:latest
docker push $MODEL_ECR_URI:latest

echo "✅ Modelo subido exitosamente a ECR"
echo "📍 URI: $MODEL_ECR_URI:latest"

# Limpiar
rm -rf temp-model-build

echo ""
echo "🎉 ¡Listo! Ahora puedes desplegar el stack con:"
echo "   sam build && sam deploy --guided" 