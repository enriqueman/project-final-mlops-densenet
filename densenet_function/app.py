'''#densenet_function\app.py
import base64
import io
import time
import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from mangum import Mangum
from typing import List, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables de entorno
AWS_REGION = os.environ.get('APP_REGION', 'us-east-1')
STAGE = os.environ.get('APP_STAGE', 'dev')
ACCOUNT_ID = os.environ.get('APP_ACCOUNT_ID', '')

# Path para el modelo (incluido en el contenedor)
MODEL_PATH = "/var/task/model/densenet121_Opset17.onnx"

# Inicializar variables globales
session = None
input_name = None
input_shape = None
output_name = None

def load_model():
    """Cargar el modelo ONNX"""
    global session, input_name, input_shape, output_name
    
    try:
        # Verificar directorios disponibles
        logger.info("=== DEBUG: Verificando estructura de directorios ===")
        logger.info(f"Contenido de /var/task: {os.listdir('/var/task') if os.path.exists('/var/task') else 'No existe'}")
        
        if os.path.exists('/var/task/model'):
            logger.info(f"Contenido de /var/task/model: {os.listdir('/var/task/model')}")
        else:
            logger.error("Directorio /var/task/model no existe")
            return False
        
        # Verificar si el modelo existe
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Modelo no encontrado en: {MODEL_PATH}")
            
            # Buscar archivos .onnx en todo el sistema
            logger.info("Buscando archivos .onnx en el contenedor...")
            import subprocess
            try:
                result = subprocess.run(['find', '/var', '-name', '*.onnx'], 
                                      capture_output=True, text=True, timeout=10)
                if result.stdout:
                    logger.info(f"Archivos .onnx encontrados: {result.stdout}")
                else:
                    logger.error("No se encontraron archivos .onnx en el contenedor")
            except Exception as e:
                logger.error(f"Error buscando archivos: {e}")
            
            return False
        
        # Verificar tamaño del archivo
        file_size = os.path.getsize(MODEL_PATH)
        logger.info(f"Tamaño del modelo: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 1000:  # Muy pequeño, probablemente no es el modelo real
            logger.error(f"El archivo del modelo es muy pequeño: {file_size} bytes")
            return False
        
        # Cargar el modelo
        logger.info(f"Cargando modelo desde: {MODEL_PATH}")
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
        logger.info("Modelo ONNX cargado exitosamente")
        
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_name = session.get_outputs()[0].name
        
        logger.info(f"Modelo cargado - Input: {input_name}, Shape: {input_shape}, Output: {output_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        session = None
        return False

# Modelos Pydantic
class ImageRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    image: str  # base64 encoded image
    top_k: Optional[int] = 5

class PredictionResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    rank: int
    class_id: int
    probability: float
    confidence: str

class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    success: bool
    predictions: List[PredictionResult]
    inference_time_ms: float
    model_info: dict

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_info: Optional[dict] = None
    debug_info: Optional[dict] = None

# Crear FastAPI app - Configuración simplificada
app = FastAPI(
    title="DenseNet121 ONNX Inference API",
    description="API para inferencia con modelo DenseNet121 en AWS Lambda",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
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
    
    debug_info = {
        "model_path": MODEL_PATH,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "aws_region": AWS_REGION,
        "stage": STAGE,
        "session_loaded": session is not None,
        "working_directory": os.getcwd(),
        "lambda_task_root": os.environ.get('LAMBDA_TASK_ROOT', 'Not set'),
        "var_task_contents": os.listdir('/var/task') if os.path.exists('/var/task') else 'No existe',
        "model_dir_contents": os.listdir('/var/task/model') if os.path.exists('/var/task/model') else 'No existe',
        "model_file_size": os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
    }
    
    # Intentar cargar el modelo si no está cargado
    if session is None:
        logger.info("Intentando cargar modelo...")
        model_loaded = load_model()
        debug_info["reload_attempt"] = model_loaded
    
    return HealthResponse(
        status="healthy" if session is not None else "unhealthy",
        model_loaded=session is not None,
        model_info={
            "name": "DenseNet121",
            "input_shape": input_shape if session else None,
            "input_name": input_name if session else None,
            "output_name": output_name if session else None,
            "parameters": 7978856,
            "source": "Included in container"
        } if session else None,
        debug_info=debug_info
    )

@app.get("/health", response_model=HealthResponse)
async def get_health():
    """Alias para health check"""
    return await health_check()

@app.post("/predict", response_model=InferenceResponse)
async def predict_image(request: ImageRequest):
    """Endpoint principal de predicción"""
    
    if session is None:
        # Intentar cargar una vez más
        if not load_model():
            raise HTTPException(
                status_code=500, 
                detail="Modelo no está cargado. Revisa logs para más detalles."
            )
    
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
        "providers": session.get_providers() if session else [],
        "opset_version": 17,
        "model_file_size": os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
    }

# Verificar que python-multipart está disponible antes de crear el endpoint
try:
    import multipart
    
    @app.post("/predict-file")
    async def predict_from_file(
        file: UploadFile = File(..., description="Imagen a procesar"),
        top_k: int = Query(default=5, description="Número de predicciones top a retornar")
    ):
        """Endpoint para subir archivo directamente"""
        
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser una imagen (JPEG, PNG, etc.)"
            )
        
        try:
            # Leer archivo y convertir a base64
            image_bytes = await file.read()
            if not image_bytes:
                raise HTTPException(
                    status_code=400,
                    detail="El archivo está vacío"
                )
                
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Usar el endpoint principal
            request = ImageRequest(image=image_b64, top_k=top_k)
            return await predict_image(request)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error procesando archivo: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error procesando archivo: {str(e)}"
            )
    
    logger.info("Endpoint predict-file habilitado")
    
except ImportError:
    logger.warning("python-multipart no está disponible, endpoint predict-file deshabilitado")

# Intentar cargar el modelo al inicio
logger.info("=== INICIALIZANDO APLICACIÓN ===")
model_loaded = load_model()
if model_loaded:
    logger.info("✅ Modelo cargado exitosamente al inicio")
else:
    logger.error("❌ Error cargando modelo al inicio")

# Crear handler para Lambda - Configuración simplificada para HTTP API Gateway
lambda_handler = Mangum(app, lifespan="off")'''

# densenet_function/app.py - VERSION ULTRA SIMPLE PARA TEST
import json
import os
import time
import logging
import traceback
from fastapi import FastAPI
from mangum import Mangum

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables de entorno básicas
STAGE = os.environ.get('APP_STAGE', 'dev')

# Crear FastAPI app ultra simple
app = FastAPI(
    title="DenseNet Test API",
    description="API de test simplificada",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {"message": "API is working", "status": "healthy"}

@app.get("/hello")
async def hello():
    """Hello World endpoint"""
    return {"message": "Hello World"}

@app.get("/test")
async def test():
    """Test endpoint"""
    return {
        "message": "Test successful",
        "stage": STAGE,
        "timestamp": time.time(),
        "status": "OK"
    }

@app.get("/ping")
async def ping():
    """Ping endpoint"""
    return "pong"

@app.get("/info")
async def info():
    """System info"""
    return {
        "app": "DenseNet Test",
        "stage": STAGE,
        "working_dir": os.getcwd(),
        "lambda_task_root": os.environ.get('LAMBDA_TASK_ROOT', 'Not set'),
        "python_version": "3.9"
    }

# Handler con debugging extenso
def lambda_handler(event, context):
    """Lambda handler con debugging"""
    logger.info("=== LAMBDA HANDLER START ===")
    logger.info(f"Event: {json.dumps(event, indent=2)}")
    logger.info(f"Context: {context}")
    
    # Información del evento
    if isinstance(event, dict):
        logger.info(f"Event keys: {list(event.keys())}")
        
        # HTTP API Gateway v2.0
        if 'version' in event and event['version'] == '2.0':
            logger.info("HTTP API Gateway v2.0 detected")
            logger.info(f"HTTP Method: {event.get('requestContext', {}).get('http', {}).get('method')}")
            logger.info(f"Path: {event.get('rawPath')}")
            
        # HTTP API Gateway v1.0 o REST API
        elif 'httpMethod' in event:
            logger.info("REST API Gateway or HTTP API v1.0 detected")
            logger.info(f"HTTP Method: {event.get('httpMethod')}")
            logger.info(f"Path: {event.get('path')}")
        
        # Otros tipos de evento
        else:
            logger.info("Unknown event type")
    
    try:
        # Crear handler Mangum
        mangum_handler = Mangum(app, lifespan="off")
        result = mangum_handler(event, context)
        
        logger.info("=== LAMBDA HANDLER SUCCESS ===")
        logger.info(f"Result: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"=== LAMBDA HANDLER ERROR ===")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Respuesta manual en caso de error
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": str(e),
                "type": "Lambda Handler Error",
                "stage": STAGE
            })
        }