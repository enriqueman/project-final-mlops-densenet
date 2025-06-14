from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import os
import logging
import time
from .model_service import ModelService
from .schemas import PredictionResponse, HealthResponse

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear instancia de FastAPI
app = FastAPI(
    title="AI Model Service",
    description="Servicio de predicción usando modelo DenseNet121",
    version="1.0.0"
)

# Inicializar servicio del modelo
model_service = None
model_loading = False
model_load_error = None

def get_model_path():
    """Determinar la ruta correcta del modelo según el entorno"""
    # Prioridad de rutas del modelo
    possible_paths = [
        os.getenv("MODEL_PATH"),  # Variable de entorno explícita
        "/code/app/model/densenet121_Opset17.onnx",  # Contenedor Docker
        "/tmp/densenet121_Opset17.onnx",  # CI/CD o tests
        "./model/densenet121_Opset17.onnx",  # Desarrollo local
        "/app/models/densenet121_Opset17.onnx"  # Fallback anterior
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            logger.info(f"Modelo encontrado en: {path}")
            return path
    
    # Si no se encuentra, usar la ruta por defecto del contenedor
    default_path = "/code/app/model/densenet121_Opset17.onnx"
    logger.warning(f"Modelo no encontrado, usando ruta por defecto: {default_path}")
    return default_path

@app.on_event("startup")
async def startup_event():
    """Inicializar el modelo al arrancar la aplicación"""
    global model_service, model_loading, model_load_error
    
    try:
        model_loading = True
        logger.info("🚀 Iniciando carga del modelo...")
        
        model_path = get_model_path()
        logger.info(f"📁 Ruta del modelo: {model_path}")
        
        # Verificar que el archivo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Archivo del modelo no encontrado: {model_path}")
        
        # Verificar tamaño del archivo
        file_size = os.path.getsize(model_path)
        logger.info(f"📊 Tamaño del modelo: {file_size / 1024 / 1024:.2f} MB")
        
        if file_size == 0:
            raise ValueError("El archivo del modelo está vacío")
        
        # Cargar modelo
        model_service = ModelService(model_path)
        model_loading = False
        logger.info("✅ Modelo cargado exitosamente")
        
    except Exception as e:
        model_loading = False
        model_load_error = str(e)
        logger.error(f"❌ Error al cargar el modelo: {e}", exc_info=True)
        # No hacer raise aquí para permitir que la app inicie y responda al health check
        # raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de salud del servicio - Compatible con ALB Health Check"""
    global model_service, model_loading, model_load_error
    
    try:
        # Si el modelo aún se está cargando
        if model_loading:
            logger.info("🔄 Modelo aún cargando...")
            return HealthResponse(
                status="loading",
                message="Modelo cargándose, espere un momento",
                model_loaded=False
            )
        
        # Si hubo error cargando el modelo
        if model_load_error:
            logger.error(f"❌ Error en carga del modelo: {model_load_error}")
            raise HTTPException(
                status_code=503, 
                detail=f"Error cargando modelo: {model_load_error}"
            )
        
        # Si no hay servicio del modelo
        if model_service is None:
            logger.error("❌ model_service es None")
            raise HTTPException(
                status_code=503, 
                detail="Modelo no inicializado"
            )
        
        # Health check básico primero
        logger.debug("🔍 Ejecutando health check básico...")
        if not model_service.simple_health_check():
            logger.error("❌ Health check básico falló")
            raise HTTPException(
                status_code=503, 
                detail="Modelo no está cargado correctamente"
            )
        
        # Para el ALB, el check básico es suficiente
        # El check completo se puede hacer en otro endpoint
        logger.info("✅ Health check exitoso")
        return HealthResponse(
            status="healthy",
            message="Servicio funcionando correctamente",
            model_loaded=True
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error inesperado en health check: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=503, 
            detail=f"Servicio no disponible: {str(e)}"
        )

@app.get("/health/ready")
async def readiness_check():
    """Check de readiness - más estricto que el health check básico"""
    global model_service, model_loading, model_load_error
    
    if model_loading:
        raise HTTPException(status_code=503, detail="Modelo aún cargando")
    
    if model_load_error:
        raise HTTPException(status_code=503, detail=f"Error: {model_load_error}")
    
    if model_service is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Test de inferencia completo
    try:
        if not model_service.is_healthy():
            raise HTTPException(status_code=503, detail="Modelo no pasa test de inferencia")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error en test: {str(e)}")
    
    return {"status": "ready", "message": "Servicio listo para recibir requests"}

@app.get("/health/live")
async def liveness_check():
    """Check de liveness - solo verifica que la app esté viva"""
    return {
        "status": "alive",
        "timestamp": time.time(),
        "app_running": True
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Endpoint para hacer predicciones"""
    try:
        logger.info(f"Recibiendo archivo: {file.filename}, tipo: {file.content_type}")
        
        if model_service is None:
            logger.error("model_service es None en predict")
            raise HTTPException(status_code=503, detail="Modelo no disponible")
        
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            logger.error(f"Tipo de archivo inválido: {file.content_type}")
            raise HTTPException(
                status_code=400, 
                detail=f"El archivo debe ser una imagen. Recibido: {file.content_type}"
            )
        
        # Leer imagen
        logger.info("Leyendo datos de imagen...")
        image_data = await file.read()
        logger.info(f"Datos leídos: {len(image_data)} bytes")
        
        try:
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"Imagen cargada: {image.size}, modo: {image.mode}")
        except Exception as img_error:
            logger.error(f"Error abriendo imagen: {img_error}")
            raise HTTPException(
                status_code=400,
                detail=f"No se pudo procesar la imagen: {str(img_error)}"
            )
        
        # Hacer predicción
        logger.info("Iniciando predicción...")
        try:
            prediction = model_service.predict(image)
            logger.info("Predicción completada exitosamente")
        except Exception as pred_error:
            logger.error(f"Error en predicción del modelo: {pred_error}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error en predicción: {str(pred_error)}"
            )
        
        return PredictionResponse(
            filename=file.filename,
            predictions=prediction["predictions"],
            confidence_scores=prediction["confidence_scores"],
            processing_time=prediction["processing_time"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inesperado en predict: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

# Mantener todos tus endpoints de debug existentes...
@app.get("/health/basic")
async def basic_health_check():
    """Health check básico - solo verifica que el servicio esté corriendo"""
    return {
        "status": "healthy",
        "message": "Servicio FastAPI funcionando",
        "timestamp": time.time(),
        "model_service_loaded": model_service is not None,
        "model_loading": model_loading,
        "model_load_error": model_load_error
    }

# ... resto de tus endpoints de debug ...