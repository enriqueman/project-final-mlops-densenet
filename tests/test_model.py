import os
import json
import boto3
import numpy as np
import onnxruntime as ort
from PIL import Image
import pytest
import requests
from io import BytesIO
import base64
import time
import botocore
import logging

# Configuración
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
STAGE = os.environ.get('STAGE', 'dev')
TEST_DATA_BUCKET = os.environ.get('TEST_DATA_BUCKET', f'densenet-test-data-{STAGE}')
MODELS_BUCKET = os.environ.get('MODELS_BUCKET', f'densenet-models-{STAGE}')
MODEL_KEY = 'test_data/test_images.json'
METRICS_THRESHOLD = {
    'accuracy': 0.25,
    'avg_confidence': 0.50
}

logger = logging.getLogger(__name__)

def wait_for_bucket(bucket_name, max_attempts=10, delay=5):
    """Esperar a que el bucket esté disponible"""
    s3 = boto3.client('s3', region_name=AWS_REGION)
    for attempt in range(max_attempts):
        try:
            s3.head_bucket(Bucket=bucket_name)
            return True
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                print(f"Bucket {bucket_name} no existe aún, esperando... (intento {attempt + 1}/{max_attempts})")
                time.sleep(delay)
            else:
                raise
    return False

def download_test_data():
    """Descargar datos de prueba desde S3"""
    # Esperar a que el bucket esté disponible
    if not wait_for_bucket(TEST_DATA_BUCKET):
        raise Exception(f"El bucket {TEST_DATA_BUCKET} no está disponible después de esperar")
    
    s3 = boto3.client('s3', region_name=AWS_REGION)
    try:
        response = s3.get_object(Bucket=TEST_DATA_BUCKET, Key=MODEL_KEY)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error descargando datos de prueba: {str(e)}")
        raise

def ensure_model_downloaded():
    """Asegurar que el modelo esté descargado localmente"""
    model_path = "/tmp/densenet121_Opset17.onnx"
    
    # Verificar si el modelo ya existe
    if os.path.exists(model_path):
        print(f"Modelo ya existe en: {model_path}")
        return model_path
    
    # Importar y ejecutar el script de descarga
    print("Descargando modelo desde S3...")
    
    # Configurar variables de entorno para el script
    os.environ['STAGE'] = STAGE
    os.environ['AWS_REGION'] = AWS_REGION
    
    try:
        # Importar el módulo de descarga
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        
        from download_model_from_s3 import download_model_from_s3
        
        if download_model_from_s3():
            if os.path.exists(model_path):
                print(f"✅ Modelo descargado exitosamente: {model_path}")
                return model_path
            else:
                raise Exception("El modelo no existe después de la descarga")
        else:
            raise Exception("Fallo en la descarga del modelo desde S3")
            
    except ImportError as e:
        print(f"Error importando script de descarga: {str(e)}")
        raise Exception("No se pudo importar el script de descarga del modelo")
    except Exception as e:
        print(f"Error descargando modelo: {str(e)}")
        raise

def preprocess_image(image_bytes):
    """Preprocesar imagen para DenseNet121"""
    image = Image.open(BytesIO(image_bytes))
    
    # Convertir a RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar
    image = image.resize((224, 224))
    
    # Normalizar
    img_array = np.array(image).astype(np.float32)  # Asegurar float32
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0  # Asegurar float32
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0   # Asegurar float32
    img_array = (img_array - mean) / std
    
    # Cambiar formato HWC → CHW y agregar batch
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Verificar tipo de datos
    assert img_array.dtype == np.float32, f"Array debe ser float32, pero es {img_array.dtype}"
    
    return img_array

def test_s3_buckets_exist():
    """Verificar que los buckets de S3 existen"""
    s3 = boto3.client('s3', region_name=AWS_REGION)
    
    # Verificar bucket de test data
    try:
        s3.head_bucket(Bucket=TEST_DATA_BUCKET)
        print(f"✅ Bucket de test data existe: {TEST_DATA_BUCKET}")
    except Exception as e:
        pytest.fail(f"Bucket de test data no existe: {TEST_DATA_BUCKET} - {str(e)}")
    
    # Verificar bucket de modelos
    try:
        s3.head_bucket(Bucket=MODELS_BUCKET)
        print(f"✅ Bucket de modelos existe: {MODELS_BUCKET}")
    except Exception as e:
        pytest.fail(f"Bucket de modelos no existe: {MODELS_BUCKET} - {str(e)}")
    
    # Verificar que el modelo existe en S3
    try:
        s3.head_object(Bucket=MODELS_BUCKET, Key='models/densenet121_Opset17.onnx')
        print(f"✅ Modelo existe en S3: s3://{MODELS_BUCKET}/models/densenet121_Opset17.onnx")
    except Exception as e:
        pytest.fail(f"Modelo no existe en S3: {str(e)}")

def test_model_download():
    """Probar descarga del modelo desde S3"""
    model_path = ensure_model_downloaded()
    
    # Verificar que el archivo existe y tiene tamaño
    assert os.path.exists(model_path), f"El modelo no existe en {model_path}"
    
    file_size = os.path.getsize(model_path)
    assert file_size > 0, "El archivo del modelo está vacío"
    
    # Verificar que es un archivo ONNX válido
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_meta = session.get_inputs()[0]
        print(f"✅ Modelo ONNX válido cargado: {input_meta.name}, shape: {input_meta.shape}")
    except Exception as e:
        pytest.fail(f"Error cargando modelo ONNX: {str(e)}")

def test_model_response():
    """Probar que el modelo responde con datos de entrada definidos"""
    # Asegurar que el modelo esté descargado
    model_path = ensure_model_downloaded()
    
    # Descargar datos de prueba
    test_data = download_test_data()
    
    # Obtener la primera imagen de prueba
    test_image = test_data['images'][0]
    image_bytes = base64.b64decode(test_image['image'])
    
    # Preprocesar imagen
    input_data = preprocess_image(image_bytes)
    
    # Verificar tipo de datos
    logger.info(f"Tipo de datos de entrada: {input_data.dtype}")
    logger.info(f"Forma de datos de entrada: {input_data.shape}")
    
    # Cargar modelo y verificar metadatos
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_meta = session.get_inputs()[0]
    logger.info(f"Nombre de entrada del modelo: {input_meta.name}")
    logger.info(f"Tipo de datos esperado: {input_meta.type}")
    logger.info(f"Forma esperada: {input_meta.shape}")
    
    # Obtener nombres de entrada/salida
    input_name = input_meta.name
    output_name = session.get_outputs()[0].name
    
    # Ejecutar inferencia
    outputs = session.run([output_name], {input_name: input_data})
    predictions = outputs[0]
    
    # Verificar formato de salida
    assert predictions.shape[1] == 1000, "El modelo debe tener 1000 clases de salida"
    assert len(predictions.shape) == 2, "La salida debe ser 2D (batch, classes)"
    assert not np.any(np.isnan(predictions)), "No debe haber valores NaN en las predicciones"

def test_model_metrics():
    """Probar que las métricas del modelo cumplen con los umbrales"""
    # Asegurar que el modelo esté descargado
    model_path = ensure_model_downloaded()
    
    # Descargar datos de prueba
    test_data = download_test_data()
    
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_meta = session.get_inputs()[0]
    
    # Obtener nombres de entrada/salida
    input_name = input_meta.name
    output_name = session.get_outputs()[0].name
    
    correct_predictions = 0
    total_confidence = 0
    
    # Evaluar cada imagen
    for test_case in test_data['images']:
        # Preprocesar imagen
        image_bytes = base64.b64decode(test_case['image'])
        input_data = preprocess_image(image_bytes)
        
        # Verificar tipo de datos
        assert input_data.dtype == np.float32, f"Datos de entrada deben ser float32, pero son {input_data.dtype}"
        
        # Ejecutar inferencia
        outputs = session.run([output_name], {input_name: input_data})
        predictions = outputs[0]
        
        # Calcular softmax
        exp_preds = np.exp(predictions - np.max(predictions))
        probabilities = exp_preds / np.sum(exp_preds)
        
        # Obtener clase predicha y confianza
        predicted_class = np.argmax(probabilities)
        confidence = np.max(probabilities)
        
        # Obtener top 5 predicciones para logging
        top5_indices = np.argsort(probabilities[0])[-5:][::-1]
        top5_probs = probabilities[0][top5_indices]
        
        logger.info(f"\nPredicciones para {test_case['filename']}:")
        logger.info(f"Clase esperada: {test_case['label']}")
        logger.info(f"Clase predicha: {predicted_class}")
        logger.info(f"Confianza: {confidence:.4f}")
        logger.info("Top 5 predicciones:")
        for idx, prob in zip(top5_indices, top5_probs):
            logger.info(f"  Clase {idx}: {prob:.4f}")
        
        # Actualizar métricas
        if predicted_class == test_case['label']:
            correct_predictions += 1
            logger.info("✓ Predicción correcta")
        else:
            logger.info("✗ Predicción incorrecta")
        total_confidence += confidence
    
    # Calcular métricas finales
    accuracy = correct_predictions / len(test_data['images'])
    avg_confidence = total_confidence / len(test_data['images'])
    
    logger.info(f"\nMétricas finales:")
    logger.info(f"Accuracy: {accuracy:.4f} (threshold: {METRICS_THRESHOLD['accuracy']})")
    logger.info(f"Avg Confidence: {avg_confidence:.4f} (threshold: {METRICS_THRESHOLD['avg_confidence']})")
    
    # Verificar umbrales
    assert accuracy >= METRICS_THRESHOLD['accuracy'], f"Accuracy {accuracy} below threshold {METRICS_THRESHOLD['accuracy']}"
    assert avg_confidence >= METRICS_THRESHOLD['avg_confidence'], f"Average confidence {avg_confidence} below threshold {METRICS_THRESHOLD['avg_confidence']}"