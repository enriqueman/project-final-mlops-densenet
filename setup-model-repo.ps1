param(
    [string]$Stage = "dev",
    [string]$AwsRegion = "us-east-1"
)

# Configuración
$ModelFile = "models/densenet121_Opset17.onnx"

# Verificar que el modelo existe
if (-not (Test-Path $ModelFile)) {
    Write-Host "Error: El archivo $ModelFile no existe" -ForegroundColor Red
    exit 1
}

Write-Host "🚀 Configurando repositorio ECR para modelos..." -ForegroundColor Green

# Obtener Account ID
$AccountId = aws sts get-caller-identity --query Account --output text
$ModelRepoName = "densenet121-model-$Stage"
$ModelEcrUri = "$AccountId.dkr.ecr.$AwsRegion.amazonaws.com/$ModelRepoName"

# Crear repositorio si no existe
Write-Host "📦 Creando repositorio ECR: $ModelRepoName" -ForegroundColor Yellow
try {
    aws ecr describe-repositories --repository-names $ModelRepoName --region $AwsRegion 2>$null
    Write-Host "Repositorio ya existe" -ForegroundColor Cyan
} catch {
    Write-Host "Creando nuevo repositorio..." -ForegroundColor Yellow
    aws ecr create-repository `
        --repository-name $ModelRepoName `
        --region $AwsRegion `
        --image-scanning-configuration scanOnPush=true `
        --image-tag-mutability MUTABLE
}

# Configurar lifecycle policy
Write-Host "⚙️  Configurando política de retención..." -ForegroundColor Yellow
$LifecyclePolicy = @'
{
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
}
'@

$LifecyclePolicy | aws ecr put-lifecycle-policy `
    --repository-name $ModelRepoName `
    --region $AwsRegion `
    --lifecycle-policy-text file:///dev/stdin

Write-Host "🔨 Construyendo imagen del modelo..." -ForegroundColor Green

# Crear directorio temporal
New-Item -ItemType Directory -Force -Path "temp-model-build"
Copy-Item $ModelFile "temp-model-build/densenet121_Opset17.onnx"

# Crear Dockerfile
$Timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
@"
FROM scratch
COPY densenet121_Opset17.onnx /model/densenet121_Opset17.onnx
LABEL model.name="densenet121"
LABEL model.version="1.0"
LABEL model.format="onnx"
LABEL build.timestamp="$Timestamp"
"@ | Out-File -FilePath "temp-model-build/Dockerfile" -Encoding ASCII

# Construir imagen
Set-Location "temp-model-build"
docker build -t "densenet121-model:latest" .
Set-Location ".."

# Login a ECR
Write-Host "🔐 Autenticando con ECR..." -ForegroundColor Yellow
$LoginToken = aws ecr get-login-password --region $AwsRegion
$LoginToken | docker login --username AWS --password-stdin $ModelEcrUri

# Tag y push
Write-Host "📤 Subiendo modelo a ECR..." -ForegroundColor Green
docker tag "densenet121-model:latest" "$ModelEcrUri`:latest"
docker push "$ModelEcrUri`:latest"

Write-Host "✅ Modelo subido exitosamente a ECR" -ForegroundColor Green
Write-Host "📍 URI: $ModelEcrUri`:latest" -ForegroundColor Cyan

# Limpiar
Remove-Item -Recurse -Force "temp-model-build"

Write-Host ""
Write-Host "🎉 ¡Listo! Ahora puedes desplegar el stack con:" -ForegroundColor Green
Write-Host "   sam build && sam deploy --guided" -ForegroundColor White 