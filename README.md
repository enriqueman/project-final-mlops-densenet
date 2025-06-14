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



# 📋 Implementación Completa: Logging de Predicciones

## ✅ **Archivos creados/modificados:**

### **1. Nuevo archivo: `app/prediction_logger.py`**
- ✅ Clase `PredictionLogger` para gestionar logs en S3
- ✅ Manejo automático de bucket y carpetas
- ✅ Formato estructurado de logs
- ✅ Métodos para leer, escribir y estadísticas

### **2. Modificado: `app/main.py`**
- ✅ Import de `PredictionLogger`
- ✅ Inicialización global del logger
- ✅ Logging automático en `/predict`
- ✅ Nuevos endpoints de gestión de logs

### **3. Modificado: CDK Stack (`cdk_fargate_deploy_stack.py`)**
- ✅ Permisos S3 para bucket `predics`
- ✅ Variable de entorno `LOGS_BUCKET`
- ✅ Output del bucket de logs

## 🎯 **Funcionalidad implementada:**

### **Logging automático:**
```python
# Cada llamada a /predict automáticamente registra:
timestamp|filename|top_prediction|confidence|processing_time|all_predictions_json
```

### **Estructura S3:**
```
predics/
├── logsdev/
│   └── predicciones_dev.txt
└── logsprod/
    └── predicciones_prod.txt
```

### **Nuevos endpoints:**
- ✅ `GET /logs/predictions` - Ver contenido de logs
- ✅ `GET /logs/predictions/download` - Descargar archivo
- ✅ `GET /logs/predictions/stats` - Estadísticas
- ✅ `DELETE /logs/predictions` - Limpiar logs (solo dev)

## 🚀 **Para desplegar:**

### **1. Asegurar que tienes todos los archivos:**
```
app/
├── main.py              (modificado)
├── prediction_logger.py (nuevo)
├── model_service.py
└── schemas.py

cdk_fargate_deploy/
└── cdk_fargate_deploy_stack.py (modificado)
```

### **2. Deploy normal:**
```bash
# La pipeline automáticamente:
# 1. Detecta los nuevos archivos
# 2. Construye imagen con PredictionLogger
# 3. Despliega con permisos S3 correctos
git push origin dev
```

### **3. Verificar funcionamiento:**
```bash
# 1. Hacer una predicción
curl -X POST "https://your-api/predict" -F "file=@test.jpg"

# 2. Verificar que se registró
curl -X GET "https://your-api/logs/predictions/stats"

# 3. Ver logs
curl -X GET "https://your-api/logs/predictions"
```

## 🔧 **Características técnicas:**

### **Tolerancia a errores:**
- ✅ Si S3 falla, predicción sigue funcionando
- ✅ Logging no bloquea la respuesta
- ✅ Manejo graceful de errores

### **Separación por entorno:**
- ✅ Dev: `logsdev/predicciones_dev.txt`
- ✅ Prod: `logsprod/predicciones_prod.txt`
- ✅ Permisos y operaciones independientes

### **Seguridad:**
- ✅ Bucket privado con permisos IAM específicos
- ✅ Solo metadatos, no imágenes
- ✅ DELETE de logs solo en dev

## 📊 **Información registrada por predicción:**

```json
{
  "timestamp": "2025-06-14T12:01:23.456789",
  "filename": "dog.jpg",
  "top_prediction": "golden_retriever", 
  "confidence": 0.8542,
  "processing_time": 0.1234,
  "all_predictions": ["golden_retriever", "labrador", "beagle", "poodle", "bulldog"],
  "confidence_scores": [0.8542, 0.1234, 0.0098, 0.0087, 0.0039]
}
```

## 🎉 **¡Listo para usar!**

Una vez desplegado, cada predicción se registrará automáticamente y podrás:

1. **Monitorear uso:** Ver cuántas predicciones se hacen
2. **Analizar rendimiento:** Tiempos de procesamiento
3. **Estudiar resultados:** Qué clases se predicen más
4. **Debugging:** Identificar predicciones problemáticas
5. **Métricas:** Confianza promedio del modelo

La implementación es **completamente backward-compatible** - no rompe funcionalidad existente y añade valor sin complejidad adicional para el usuario final.