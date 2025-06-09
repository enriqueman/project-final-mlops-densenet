param(
    [string]$Stage = "dev",
    [string]$AwsRegion = "us-east-1"
)

# Validar el Stage
if ($Stage -notin @("dev", "prod")) {
    Write-Host "❌ Error: El stage debe ser 'dev' o 'prod'" -ForegroundColor Red
    exit 1
}

# Configuración
$ModelFile = "models/densenet121_Opset17.onnx"
$StatsFile = "models/turnkey_stats.yaml"

# Verificar que los archivos existen
if (-not (Test-Path $ModelFile)) {
    Write-Host "❌ Error: El archivo $ModelFile no existe" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $StatsFile)) {
    Write-Host "⚠️ Advertencia: El archivo $StatsFile no existe" -ForegroundColor Yellow
}

Write-Host "🚀 Configurando repositorio ECR para modelos ($Stage)..." -ForegroundColor Green

# Verificar Docker
try {
    docker info > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error: Docker no está corriendo" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Error: Docker no está instalado o no está accesible" -ForegroundColor Red
    exit 1
}

# Verificar AWS CLI y credenciales
try {
    $AccountId = aws sts get-caller-identity --query Account --output text 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error: No se pudo obtener el Account ID. Verifica tus credenciales de AWS" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Error: AWS CLI no está instalado o las credenciales no están configuradas" -ForegroundColor Red
    exit 1
}

$ModelRepoName = "densenet121-model-$Stage"
$ModelEcrUri = "$AccountId.dkr.ecr.$AwsRegion.amazonaws.com/$ModelRepoName"

# Crear repositorio si no existe
Write-Host "📦 Creando/Verificando repositorio ECR: $ModelRepoName" -ForegroundColor Yellow
try {
    aws ecr describe-repositories --repository-names $ModelRepoName --region $AwsRegion 2>$null
    Write-Host "✅ Repositorio ya existe" -ForegroundColor Cyan
} catch {
    Write-Host "🆕 Creando nuevo repositorio..." -ForegroundColor Yellow
    try {
        aws ecr create-repository `
            --repository-name $ModelRepoName `
            --region $AwsRegion `
            --image-scanning-configuration scanOnPush=true `
            --image-tag-mutability MUTABLE
        
        if ($LASTEXITCODE -ne 0) { throw }
        Write-Host "✅ Repositorio creado exitosamente" -ForegroundColor Green
    } catch {
        Write-Host "❌ Error al crear el repositorio" -ForegroundColor Red
        exit 1
    }
}

# Configurar lifecycle policy
Write-Host "⚙️  Configurando política de retención..." -ForegroundColor Yellow
$LifecyclePolicy = @{
    rules = @(
        @{
            rulePriority = 1
            description = "Keep last 3 model versions"
            selection = @{
                tagStatus = "any"
                countType = "imageCountMoreThan"
                countNumber = 3
            }
            action = @{
                type = "expire"
            }
        }
    )
} | ConvertTo-Json -Depth 10

try {
    aws ecr put-lifecycle-policy `
        --repository-name $ModelRepoName `
        --region $AwsRegion `
        --lifecycle-policy-text $LifecyclePolicy

    if ($LASTEXITCODE -ne 0) { throw }
    Write-Host "✅ Política de retención configurada" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Error al configurar la política de retención" -ForegroundColor Yellow
}

Write-Host "🔨 Construyendo imagen del modelo..." -ForegroundColor Green

# Crear directorio temporal y copiar archivos necesarios
Write-Host "📂 Preparando archivos..." -ForegroundColor Yellow
$TempDir = "temp-model-build-$Stage"
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
Copy-Item $ModelFile "$TempDir/densenet121_Opset17.onnx"
Copy-Item "model-image/Dockerfile" "$TempDir/Dockerfile"

if (Test-Path $StatsFile) {
    Copy-Item $StatsFile "$TempDir/turnkey_stats.yaml"
}

# Autenticar con ECR
Write-Host "🔐 Autenticando con ECR..." -ForegroundColor Yellow
try {
    aws ecr get-login-password --region $AwsRegion | docker login --username AWS --password-stdin $ModelEcrUri
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "❌ Error al autenticar con ECR" -ForegroundColor Red
    Remove-Item -Recurse -Force $TempDir
    exit 1
}

# Construir imagen
Write-Host "🏗️ Construyendo imagen Docker..." -ForegroundColor Yellow
Set-Location $TempDir
try {
    docker build -t "densenet121-model-$Stage`:latest" .
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "❌ Error al construir la imagen Docker" -ForegroundColor Red
    Set-Location ".."
    Remove-Item -Recurse -Force $TempDir
    exit 1
}
Set-Location ".."

# Tag y push
Write-Host "📤 Subiendo imagen a ECR..." -ForegroundColor Green
try {
    docker tag "densenet121-model-$Stage`:latest" "$ModelEcrUri`:latest"
    docker push "$ModelEcrUri`:latest"
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "❌ Error al subir la imagen a ECR" -ForegroundColor Red
    Remove-Item -Recurse -Force $TempDir
    exit 1
}

# Verificar que la imagen se subió correctamente
Write-Host "✅ Verificando imagen en ECR..." -ForegroundColor Yellow
try {
    $images = aws ecr describe-images --repository-name $ModelRepoName --region $AwsRegion 2>$null
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "❌ Error al verificar la imagen en ECR" -ForegroundColor Red
    Remove-Item -Recurse -Force $TempDir
    exit 1
}

# Limpiar
Write-Host "🧹 Limpiando archivos temporales..." -ForegroundColor Yellow
Remove-Item -Recurse -Force $TempDir

Write-Host ""
Write-Host "✅ Modelo subido exitosamente a ECR ($Stage)" -ForegroundColor Green
Write-Host "📍 URI: $ModelEcrUri`:latest" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎉 ¡Listo! Ahora puedes desplegar el stack con:" -ForegroundColor Green
Write-Host "   sam build; sam deploy --guided" -ForegroundColor "White" 