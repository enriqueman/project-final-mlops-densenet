from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import os
import logging
import time
import traceback

# CAMBIO: Import absoluto en lugar de relativo
try:
    from model_service import ModelService
    from schemas import PredictionResponse, HealthResponse
except ImportError:
    # Fallback para imports relativos si es necesario
    try:
        from .model_service import ModelService
        from .schemas import PredictionResponse, HealthResponse
    except ImportError:
        # Último fallback - crear clases básicas inline
        from pydantic import BaseModel
        from typing import List, Optional
        
        class HealthResponse(BaseModel):
            status: str
            message: str
            model_loaded: bool
            
        class PredictionResponse(BaseModel):
            filename: Optional[str]
            predictions: List[str]
            confidence_scores: List[float]
            processing_time: float

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear instancia de FastAPI
app = FastAPI(
    title="AI Model Service",
    description="Servicio de predicción usando modelo DenseNet121",
    version="1.0.0"
)

# Variables globales para el estado del modelo
model_service = None
model_loading = True
model_load_error = None
startup_time = time.time()

def get_model_path():
    """Determinar la ruta correcta del modelo según el entorno"""
    possible_paths = [
        os.getenv("MODEL_PATH"),
        "/code/app/model/densenet121_Opset17.onnx",  # Contenedor cloud
        "/app/models/densenet121_Opset17.onnx",      # Contenedor local  
        "/tmp/densenet121_Opset17.onnx",             # CI/CD
        "./model/densenet121_Opset17.onnx"           # Desarrollo local
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            logger.info(f"✅ Modelo encontrado en: {path}")
            return path
    
    # Mostrar rutas buscadas para debugging
    logger.error("❌ Modelo no encontrado en ninguna ruta:")
    for path in possible_paths:
        if path:
            exists = "EXISTS" if os.path.exists(path) else "NOT FOUND"
            logger.error(f"   {path} - {exists}")
    
    return possible_paths[1]  # Usar la ruta por defecto del contenedor

@app.on_event("startup")
async def startup_event():
    """Inicializar el modelo al arrancar la aplicación"""
    global model_service, model_loading, model_load_error
    
    logger.info("🚀 Iniciando aplicación FastAPI...")
    logger.info(f"📊 Variables de entorno:")
    logger.info(f"   STAGE: {os.getenv('STAGE')}")
    logger.info(f"   AWS_REGION: {os.getenv('AWS_REGION')}")
    logger.info(f"   MODEL_PATH: {os.getenv('MODEL_PATH')}")
    
    try:
        model_loading = True
        model_path = get_model_path()
        logger.info(f"📁 Intentando cargar modelo desde: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Archivo del modelo no encontrado: {model_path}")
        
        # Verificar tamaño del archivo
        file_size = os.path.getsize(model_path)
        logger.info(f"📊 Tamaño del modelo: {file_size / 1024 / 1024:.2f} MB")
        
        if file_size == 0:
            raise ValueError("El archivo del modelo está vacío")
        
        # Intentar importar ModelService solo cuando lo necesitemos
        try:
            from model_service import ModelService
        except ImportError:
            logger.warning("No se pudo importar ModelService, continuando sin modelo...")
            model_service = None
            model_loading = False
            return
            
        model_service = ModelService(model_path)
        model_loading = False
        logger.info("✅ Modelo cargado exitosamente")
        
    except Exception as e:
        model_loading = False
        model_load_error = str(e)
        logger.error(f"❌ Error al cargar el modelo: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        # No hacer raise para permitir que la app responda al health check

@app.get("/")
async def root():
    """Endpoint raíz para health check del ALB"""
    return {"mensaje": "API  - Bienvenido"}

@app.get("/health")
async def health_check():
    """Health check principal"""
    # Simplemente redirigir al endpoint raíz
    return await root()

@app.get("/health/basic")
async def basic_health_check():
    """Health check básico"""
    return {
        "status": "healthy",
        "message": "Servicio FastAPI funcionando",
        "timestamp": time.time(),
        "model_service_loaded": model_service is not None,
        "model_loading": model_loading,
        "model_load_error": model_load_error
    }

@app.get("/debug")
async def debug_info():
    """Información de debug"""
    return {
        "app_status": "running",
        "python_path": os.getcwd(),
        "environment_variables": {
            "STAGE": os.getenv("STAGE"),
            "AWS_REGION": os.getenv("AWS_REGION"), 
            "MODEL_PATH": os.getenv("MODEL_PATH"),
        },
        "file_system": {
            "cwd": os.getcwd(),
            "exists_code_app": os.path.exists("/code/app"),
            "exists_code_app_model": os.path.exists("/code/app/model"),
            "exists_tmp": os.path.exists("/tmp"),
        },
        "uptime": time.time() - startup_time,
        "model_service": model_service is not None,
        "model_loading": model_loading,
        "model_load_error": model_load_error
    }

# Solo añadir endpoints de predicción si tenemos ModelService disponible
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint para hacer predicciones"""
    try:
        logger.info(f"Recibiendo archivo: {file.filename}, tipo: {file.content_type}")
        
        if model_service is None:
            logger.error("model_service es None en predict")
            raise HTTPException(status_code=503, detail="Modelo no disponible - servicio en modo degradado")
        
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

@app.get("/model/info")
async def model_info():
    """Información del modelo"""
    if model_service is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    return model_service.get_model_info()

@app.get("/health/detailed")
async def detailed_health_check():
    """Health check detallado con múltiples verificaciones"""
    try:
        if model_service is None:
            return {
                "status": "unhealthy",
                "message": "Modelo no inicializado",
                "checks": {
                    "model_service": False,
                    "simple_check": False,
                    "inference_check": False
                }
            }
        
        checks = {}
        
        # Check 1: Servicio básico
        checks["model_service"] = True
        
        # Check 2: Verificación simple
        try:
            checks["simple_check"] = model_service.simple_health_check()
        except Exception as e:
            logger.error(f"Simple check falló: {e}")
            checks["simple_check"] = False
        
        # Check 3: Verificación de inferencia
        try:
            checks["inference_check"] = model_service.is_healthy()
        except Exception as e:
            logger.error(f"Inference check falló: {e}")
            checks["inference_check"] = False
        
        # Determinar estado general
        if all(checks.values()):
            status = "healthy"
            message = "Todos los checks pasaron"
        elif checks.get("simple_check", False):
            status = "degraded"
            message = "Modelo cargado pero problemas en inferencia"
        else:
            status = "unhealthy"
            message = "Problemas críticos detectados"
        
        return {
            "status": status,
            "message": message,
            "checks": checks,
            "model_info": model_service.get_model_info() if model_service else None
        }
        
    except Exception as e:
        logger.error(f"Error en detailed health check: {e}")
        return {
            "status": "unhealthy",
            "message": f"Error en health check: {str(e)}",
            "checks": {}
        }

@app.get("/debug/test-preprocessing")
async def debug_test_preprocessing():
    """Endpoint para probar el preprocesamiento paso a paso"""
    try:
        if model_service is None:
            return {"error": "model_service es None"}
        
        logger.info("Iniciando test de preprocesamiento...")
        
        # Crear diferentes tipos de imágenes de prueba
        test_cases = [
            ("RGB_red", Image.new('RGB', (224, 224), color=(255, 0, 0))),
            ("RGB_gradient", Image.new('RGB', (100, 100), color=(128, 128, 128))),
            ("RGBA_with_alpha", Image.new('RGBA', (150, 150), color=(0, 255, 0, 128))),
        ]
        
        results = {}
        
        for case_name, test_image in test_cases:
            try:
                logger.info(f"Probando caso: {case_name}")
                
                # Preprocesar la imagen
                processed = model_service._preprocess_image(test_image)
                
                results[case_name] = {
                    "status": "success",
                    "original_size": test_image.size,
                    "original_mode": test_image.mode,
                    "processed_shape": processed.shape,
                    "processed_dtype": str(processed.dtype),
                    "min_value": float(processed.min()),
                    "max_value": float(processed.max()),
                    "mean_value": float(processed.mean())
                }
                
                logger.info(f"Caso {case_name}: ✅ Exitoso")
                
            except Exception as e:
                logger.error(f"Caso {case_name} falló: {e}")
                results[case_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return {
            "test_results": results,
            "expected_input_shape": [1, 3, 224, 224],
            "expected_dtype": "float32"
        }
        
    except Exception as e:
        logger.error(f"Error en test de preprocesamiento: {e}")
        return {"error": f"Error general: {str(e)}"}

@app.get("/debug/predict-test")
async def debug_predict_test():
    """Endpoint para debuggear predicciones con imagen de prueba"""
    try:
        if model_service is None:
            return {"error": "model_service es None"}
        
        logger.info("Iniciando debug predict test...")
        
        # Crear imagen de prueba
        test_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        logger.info(f"Imagen de prueba creada: {test_image.size}, modo: {test_image.mode}")
        
        # Intentar predicción paso a paso
        try:
            # Paso 1: Preprocesamiento
            logger.info("Paso 1: Preprocesamiento...")
            input_data = model_service._preprocess_image(test_image)
            logger.info(f"Preprocesamiento exitoso. Shape: {input_data.shape}")
            
            # Paso 2: Inferencia
            logger.info("Paso 2: Inferencia...")
            outputs = model_service.session.run(
                [model_service.output_name], 
                {model_service.input_name: input_data}
            )
            logger.info(f"Inferencia exitosa. Output shape: {outputs[0].shape}")
            
            # Paso 3: Post-procesamiento
            logger.info("Paso 3: Post-procesamiento...")
            predictions = outputs[0][0]
            logger.info(f"Predictions shape: {predictions.shape}")
            logger.info(f"Predictions dtype: {predictions.dtype}")
            logger.info(f"Predictions min/max: {predictions.min()}/{predictions.max()}")
            
            # Verificar softmax
            exp_predictions = np.exp(predictions - np.max(predictions))
            probabilities = exp_predictions / np.sum(exp_predictions)
            logger.info(f"Probabilities sum: {probabilities.sum()}")
            
            # Top 5
            top_indices = np.argsort(probabilities)[-5:][::-1]
            logger.info(f"Top indices: {top_indices}")
            
            return {
                "status": "success",
                "steps_completed": [
                    "image_creation",
                    "preprocessing", 
                    "inference",
                    "postprocessing"
                ],
                "input_shape": input_data.shape,
                "output_shape": outputs[0].shape,
                "predictions_shape": predictions.shape,
                "top_indices": top_indices.tolist(),
                "probabilities_sum": float(probabilities.sum())
            }
            
        except Exception as step_error:
            logger.error(f"Error en debug predict: {step_error}", exc_info=True)
            return {
                "status": "error",
                "error": str(step_error),
                "error_type": type(step_error).__name__
            }
            
    except Exception as e:
        logger.error(f"Error general en debug predict: {e}", exc_info=True)
        return {"error": f"Error general: {str(e)}"}

@app.get("/debug/health")
async def debug_health():
    """Endpoint de debug para el health check"""
    try:
        if model_service is None:
            return {"error": "model_service es None"}
        
        # Información detallada del modelo
        model_info = model_service.get_model_info()
        
        # Intentar health check con detalles
        try:
            is_healthy = model_service.is_healthy()
            return {
                "model_loaded": True,
                "is_healthy": is_healthy,
                "model_info": model_info,
                "session_providers": model_service.session.get_providers() if model_service.session else None
            }
        except Exception as health_error:
            return {
                "model_loaded": True,
                "is_healthy": False,
                "health_error": str(health_error),
                "model_info": model_info
            }
    except Exception as e:
        return {"error": f"Error general: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)