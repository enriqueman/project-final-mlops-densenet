import base64
import io
import time
import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum
from typing import List, Optional
import logging
import boto3
import docker
import tempfile

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables de entorno para ECR
AWS_REGION = os.environ.get('APP_REGION', 'us-east-1')
STAGE = os.environ.get('APP_STAGE', 'dev')
ACCOUNT_ID = os.environ.get('APP_ACCOUNT_ID', '')
MODEL_ECR_REPOSITORY = os.environ.get('MODEL_ECR_REPOSITORY', f'densenet121-model-{STAGE}')

# Path temporal para el modelo
MODEL_PATH = "/tmp/densenet121_Opset17.onnx"

def get_account_id():
    """Obtener Account ID de AWS"""
    try:
        sts_client = boto3.client('sts', region_name=AWS_REGION)
        return sts_client.get_caller_identity()['Account']
    except Exception as e:
        logger.error(f"Error obteniendo Account ID: {str(e)}")
        return None

def download_model_from_ecr():
    """Descargar modelo desde ECR"""
    try:
        logger.info("Descargando modelo desde ECR...")
        
        # Obtener Account ID si no está disponible
        account_id = ACCOUNT_ID or get_account_id()
        if not account_id:
            logger.error("No se pudo obtener el Account ID")
            return False
        
        # Configurar ECR
        ecr_client = boto3.client('ecr', region_name=AWS_REGION)
        
        # Construir URI del modelo
        model_image_uri = f"{account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com/{MODEL_ECR_REPOSITORY}:latest"
        
        logger.info(f"Descargando imagen: {model_image_uri}")
        
        # Obtener token de autorización para Docker
        token_response = ecr_client.get_authorization_token()
        token = token_response['authorizationData'][0]['authorizationToken']
        registry_url = token_response['authorizationData'][0]['proxyEndpoint']
        
        # Decodificar token
        username, password = base64.b64decode(token).decode().split(':')
        
        # Usar el cliente Docker de Python
        try:
            docker_client = docker.from_env()
            
            # Login
            docker_client.login(
                username=username,
                password=password,
                registry=registry_url
            )
            
            # Pull de la imagen
            logger.info("Descargando imagen del modelo...")
            image = docker_client.images.pull(model_image_uri)
            
            # Crear un contenedor temporal para extraer el archivo
            container = docker_client.containers.create(
                image=model_image_uri,
                command="true"  # comando dummy
            )
            
            # Extraer el archivo del contenedor
            logger.info("Extrayendo modelo del contenedor...")
            
            # Obtener el archivo como tar stream
            stream, _ = container.get_archive('/model/densenet121_Opset17.onnx')
            
            # Escribir el archivo
            import tarfile
            with tempfile.NamedTemporaryFile() as tmp_tar:
                for chunk in stream:
                    tmp_tar.write(chunk)
                tmp_tar.flush()
                
                # Extraer del tar
                with tarfile.open(tmp_tar.name, 'r') as tar:
                    tar.extractall('/tmp/')
                    
            # El archivo ahora debería estar en /tmp/densenet121_Opset17.onnx
            if os.path.exists('/tmp/densenet121_Opset17.onnx'):
                # Mover al path esperado
                os.rename('/tmp/densenet121_Opset17.onnx', MODEL_PATH)
                logger.info("Modelo extraído exitosamente")
            
            # Limpiar contenedor
            container.remove()
            
            return True
            
        except Exception as docker_error:
            logger.error(f"Error con Docker: {str(docker_error)}")
            return False
        
    except Exception as e:
        logger.error(f"Error descargando modelo desde ECR: {str(e)}")
        return False

# Inicializar sesión como None
session = None

def load_model():
    """Cargar el modelo ONNX"""
    global session, input_name, input_shape, output_name
    
    try:
        # Si el modelo no existe, intentar descargarlo
        if not os.path.exists(MODEL_PATH):
            if not download_model_from_ecr():
                logger.error("No se pudo descargar el modelo")
                return False
        
        # Cargar el modelo
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
        logger.info("Modelo ONNX cargado exitosamente")
        
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_name = session.get_outputs()[0].name
        
        return True
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        session = None
        return False

# Intentar cargar el modelo al inicio
model_loaded = load_model()

# Modelos Pydantic
class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    top_k: Optional[int] = 5

class PredictionResult(BaseModel):
    rank: int
    class_id: int
    probability: float
    confidence: str

class InferenceResponse(BaseModel):
    success: bool
    predictions: List[PredictionResult]
    inference_time_ms: float
    model_info: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[dict] = None

# Crear FastAPI app
app = FastAPI(
    title="DenseNet121 ONNX Inference API",
    description="API para inferencia con modelo DenseNet121 en AWS Lambda",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_data: str) -> np.ndarray:
    """Preprocesar imagen para DenseNet121"""
    try:
        # Decodificar desde base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convertir a RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionar
        image = image.resize((224, 224))
        
        # Normalizar
        img_array = np.array(image).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406]) * 255.0
        std = np.array([0.229, 0.224, 0.225]) * 255.0
        img_array = (img_array - mean) / std
        
        # Cambiar formato HWC → CHW y agregar batch
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Endpoint de salud"""
    global session
    
    # Intentar cargar el modelo si no está cargado
    if session is None:
        model_loaded = load_model()
    
    return HealthResponse(
        status="healthy" if session is not None else "unhealthy",
        model_loaded=session is not None,
        model_info={
            "name": "DenseNet121",
            "input_shape": input_shape if session else None,
            "parameters": 7978856,
            "source": "Downloaded from ECR"
        } if session else None
    )

@app.get("/health", response_model=HealthResponse)
async def get_health():
    """Alias para health check"""
    return await health_check()

@app.post("/predict", response_model=InferenceResponse)
async def predict_image(request: ImageRequest):
    """Endpoint principal de predicción"""
    
    if session is None:
        raise HTTPException(status_code=500, detail="Modelo no está cargado")
    
    try:
        # Preprocesar imagen
        start_time = time.time()
        processed_image = preprocess_image(request.image)
        
        # Ejecutar inferencia
        outputs = session.run([output_name], {input_name: processed_image})
        predictions = outputs[0]
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # convertir a ms
        
        # Procesar resultados
        exp_predictions = np.exp(predictions - np.max(predictions))
        probabilities = exp_predictions / np.sum(exp_predictions)
        
        # Top K predicciones
        top_k = min(request.top_k, len(probabilities[0]))
        top_indices = np.argsort(probabilities[0])[-top_k:][::-1]
        top_probs = probabilities[0][top_indices]
        
        results = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            results.append(PredictionResult(
                rank=i + 1,
                class_id=int(idx),
                probability=float(prob),
                confidence=f"{prob * 100:.2f}%"
            ))
        
        return InferenceResponse(
            success=True,
            predictions=results,
            inference_time_ms=inference_time,
            model_info={
                "name": "DenseNet121",
                "input_shape": input_shape,
                "parameters": 7978856
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/predict-file")
async def predict_from_file(file: UploadFile = File(...), top_k: int = 5):
    """Endpoint para subir archivo directamente"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer archivo y convertir a base64
        image_bytes = await file.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Usar el endpoint principal
        request = ImageRequest(image=image_b64, top_k=top_k)
        return await predict_image(request)
        
    except Exception as e:
        logger.error(f"Error procesando archivo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Información detallada del modelo"""
    if session is None:
        raise HTTPException(status_code=500, detail="Modelo no está cargado")
    
    return {
        "name": "DenseNet121",
        "framework": "ONNX",
        "input_shape": input_shape,
        "input_name": input_name,
        "output_name": output_name,
        "parameters": 7978856,
        "providers": session.get_providers(),
        "opset_version": 17
    }

# Crear handler para Lambda usando Mangum
lambda_handler = Mangum(app, lifespan="off")