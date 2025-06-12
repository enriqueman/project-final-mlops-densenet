# app.py para EC2
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
from typing import List, Optional
import logging
import uvicorn
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/app/densenet.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Variables de entorno
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
STAGE = os.environ.get('STAGE', 'dev')
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/model/densenet121_Opset17.onnx')

# Inicializar variables globales
session = None
input_name = None
input_shape = None
output_name = None

def load_model():
    """Cargar el modelo ONNX"""
    global session, input_name, input_shape, output_name
    
    try:
        logger.info("=== CARGANDO MODELO ===")
        logger.info(f"Ruta del modelo: {MODEL_PATH}")
        
        # Verificar si el modelo existe
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Modelo no encontrado en: {MODEL_PATH}")
            logger.info(f"Contenido del directorio /app/model: {os.listdir('/app/model') if os.path.exists('/app/model') else 'No existe'}")
            return False
        
        # Verificar tamaño del archivo
        file_size = os.path.getsize(MODEL_PATH)
        logger.info(f"Tamaño del modelo: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 1000:  # Muy pequeño, probablemente no es el modelo real
            logger.error(f"El archivo del modelo es muy pequeño: {file_size} bytes")
            return False
        
        # Cargar el modelo
        logger.info(f"Cargando modelo ONNX desde: {MODEL_PATH}")
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
    system_info: Optional[dict] = None

# Crear FastAPI app
app = FastAPI(
    title="DenseNet121 ONNX Inference API on EC2",
    description="API para inferencia con modelo DenseNet121 en instancia EC2",
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
    
    system_info = {
        "model_path": MODEL_PATH,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "aws_region": AWS_REGION,
        "stage": STAGE,
        "session_loaded": session is not None,
        "working_directory": os.getcwd(),
        "python_version": sys.version,
        "model_file_size": os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
    }
    
    # Intentar cargar el modelo si no está cargado
    if session is None:
        logger.info("Intentando cargar modelo...")
        model_loaded = load_model()
        system_info["reload_attempt"] = model_loaded
    
    return HealthResponse(
        status="healthy" if session is not None else "unhealthy",
        model_loaded=session is not None,
        model_info={
            "name": "DenseNet121",
            "input_shape": input_shape if session else None,
            "input_name": input_name if session else None,
            "output_name": output_name if session else None,
            "parameters": 7978856,
            "source": "Downloaded from ECR model repository"
        } if session else None,
        system_info=system_info
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