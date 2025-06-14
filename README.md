# DenseNet121 Lambda Inference API

## Prerequisitos

- AWS CLI configurado
- Docker instalado
- SAM CLI instalado
- Modelo ONNX en `models/densenet121_Opset17.onnx`

## Deployment Manual

### Paso 1: Configurar modelo en ECR (Solo la primera vez)

```bash
# Linux/Mac
chmod +x setup-model-repo.sh
./setup-model-repo.sh dev us-east-1

# Windows PowerShell
.\setup-model-repo.ps1 -Stage dev -AwsRegion us-east-1
```

### Paso 2: Desplegar infraestructura

```bash
sam build
sam deploy --guided
```

## GitHub Actions

La infraestructura está lista para ser desplegada via GitHub Actions. El modelo debe ser configurado manualmente una sola vez.

### Variables necesarias para GitHub Actions:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `STAGE`

## Model Bucket Setup

The `setup-models-bucket.sh` script is used to create and configure an S3 bucket for storing model files. This script handles both development and production environments.

### Features

- Creates a dedicated S3 bucket for models (`densenet-models-dev` or `densenet-models-prod`)
- Configures bucket settings:
  - Enables versioning for model version control
  - Sets up lifecycle policy to manage old versions
  - Blocks public access for security
- Creates a `models` directory in the bucket
- Uploads all models from the local `./models` directory

### Usage

```bash
# For development environment (default)
./setup-models-bucket.sh

# For production environment
./setup-models-bucket.sh prod

# Specify custom AWS region
./setup-models-bucket.sh dev us-west-2
```

### Prerequisites

1. AWS CLI installed and configured with valid credentials
2. Models stored in the local `./models` directory
3. Script execution permissions:
   ```bash
   chmod +x setup-models-bucket.sh
   ```

### Bucket Structure

```
densenet-models-{stage}/
└── models/
    └── [your model files]
```
