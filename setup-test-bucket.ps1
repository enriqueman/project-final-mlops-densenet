param(
    [string]$Stage = "dev",
    [string]$AwsRegion = "us-east-1"
)

# Validar el Stage
if ($Stage -notin @("dev", "prod")) {
    Write-Host "Error: El stage debe ser 'dev' o 'prod'" -ForegroundColor Red
    exit 1
}

# Configuración
$BucketName = "densenet-test-data-$Stage"
$TestDataDir = "test_data"
$TestImagesDir = "test_data/images"
$TestLabelsFile = "test_data/imagenet_labels.txt"

# Verificar AWS CLI y credenciales
try {
    Write-Host "Verificando credenciales AWS..." -ForegroundColor Yellow
    $AccountId = aws sts get-caller-identity --query Account --output text 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: No se pudo obtener el Account ID. Verifica tus credenciales de AWS" -ForegroundColor Red
        exit 1
    }
    Write-Host "Credenciales verificadas. Account ID: $AccountId" -ForegroundColor Green
} catch {
    Write-Host "Error: AWS CLI no está instalado o las credenciales no están configuradas" -ForegroundColor Red
    exit 1
}

Write-Host "Configurando bucket S3 para datos de prueba ($Stage)..." -ForegroundColor Green

# Crear o verificar bucket
Write-Host "Verificando/Creando bucket S3: $BucketName" -ForegroundColor Yellow

# Intentar crear el bucket
try {
    # Verificar si el bucket existe usando aws s3 ls
    $bucketExists = $false
    $bucketList = aws s3 ls "s3://$BucketName" 2>&1
    if ($LASTEXITCODE -eq 0) {
        $bucketExists = $true
        Write-Host "El bucket existe y tienes acceso" -ForegroundColor Green
    } else {
        Write-Host "El bucket no existe o no tienes acceso, intentando crearlo..." -ForegroundColor Yellow
    }

    if (-not $bucketExists) {
        # Crear el bucket con la ubicación específica
        Write-Host "Creando bucket en la región $AwsRegion..." -ForegroundColor Yellow
        if ($AwsRegion -eq "us-east-1") {
            aws s3api create-bucket `
                --bucket $BucketName `
                --region $AwsRegion
        } else {
            aws s3api create-bucket `
                --bucket $BucketName `
                --region $AwsRegion `
                --create-bucket-configuration LocationConstraint=$AwsRegion
        }

        if ($LASTEXITCODE -ne 0) { 
            Write-Host "Error al crear el bucket" -ForegroundColor Red
            exit 1 
        }
        Write-Host "Bucket creado exitosamente" -ForegroundColor Green

        # Esperar unos segundos para que el bucket esté disponible
        Write-Host "Esperando a que el bucket esté disponible..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10

        # Habilitar versionamiento
        Write-Host "Configurando versionamiento..." -ForegroundColor Yellow
        aws s3api put-bucket-versioning `
            --bucket $BucketName `
            --versioning-configuration Status=Enabled

        # Bloquear acceso público
        Write-Host "Configurando acceso público..." -ForegroundColor Yellow
        aws s3api put-public-access-block `
            --bucket $BucketName `
            --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
    }

    # Verificar que el bucket está accesible
    Write-Host "Verificando acceso al bucket..." -ForegroundColor Yellow
    $maxAttempts = 5
    $attempt = 1
    $success = $false

    while ($attempt -le $maxAttempts -and -not $success) {
        Write-Host "Intento $attempt de $maxAttempts..." -ForegroundColor Yellow
        try {
            aws s3 ls "s3://$BucketName" > $null
            if ($LASTEXITCODE -eq 0) {
                $success = $true
                Write-Host "Acceso al bucket verificado exitosamente" -ForegroundColor Green
            } else {
                throw
            }
        } catch {
            if ($attempt -eq $maxAttempts) {
                Write-Host "Error: No se pudo verificar el acceso al bucket después de $maxAttempts intentos" -ForegroundColor Red
                throw
            }
            Write-Host "Esperando antes de reintentar..." -ForegroundColor Yellow
            Start-Sleep -Seconds 5
            $attempt++
        }
    }

    # Crear estructura de directorios
    Write-Host "Creando estructura de directorios..." -ForegroundColor Yellow

    # Crear directorios necesarios
    Write-Host "Creando directorio de datos de prueba..." -ForegroundColor Yellow
    aws s3api put-object --bucket $BucketName --key "$TestDataDir/"
    if ($LASTEXITCODE -ne 0) { throw }

    Write-Host "Creando directorio de imágenes..." -ForegroundColor Yellow
    aws s3api put-object --bucket $BucketName --key "$TestImagesDir/"
    if ($LASTEXITCODE -ne 0) { throw }

    # Generar y subir datos de prueba
    Write-Host "Instalando dependencias de Python..." -ForegroundColor Yellow
    python -m pip install -r scripts/requirements.txt
    if ($LASTEXITCODE -ne 0) { 
        Write-Host "Error al instalar dependencias de Python" -ForegroundColor Red
        throw 
    }

    Write-Host "Generando y subiendo datos de prueba..." -ForegroundColor Yellow
    $env:STAGE = $Stage
    $env:AWS_REGION = $AwsRegion
    python scripts/generate_test_data.py
    if ($LASTEXITCODE -ne 0) { 
        Write-Host "Error al generar y subir datos de prueba" -ForegroundColor Red
        throw 
    }

    Write-Host ""
    Write-Host "Configuración completada exitosamente" -ForegroundColor Green
    Write-Host "Bucket: $BucketName" -ForegroundColor Cyan
    Write-Host "Región: $AwsRegion" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Estructura del bucket:" -ForegroundColor Yellow
    aws s3 ls "s3://$BucketName" --recursive

} catch {
    Write-Host "Error durante la configuración del bucket: $_" -ForegroundColor Red
    Write-Host "Detalles adicionales del error:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
} 