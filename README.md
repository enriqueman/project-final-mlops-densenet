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
