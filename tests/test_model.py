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

# Configuración
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
STAGE = os.environ.get('STAGE', 'dev')
MODEL_BUCKET = os.environ.get('TEST_DATA_BUCKET', f'densenet-test-data-{STAGE}')
MODEL_KEY = 'test_data/test_images.json'
METRICS_THRESHOLD = {
    'accuracy': 0.85,
    'avg_confidence': 0.70
}

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
    if not wait_for_bucket(MODEL_BUCKET):
        raise Exception(f"El bucket {MODEL_BUCKET} no está disponible después de esperar")
    
    s3 = boto3.client('s3', region_name=AWS_REGION)
    try:
        response = s3.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error descargando datos de prueba: {str(e)}")
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
    img_array = np.array(image).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    std = np.array([0.229, 0.224, 0.225]) * 255.0
    img_array = (img_array - mean) / std
    
    # Cambiar formato HWC → CHW y agregar batch
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def test_model_response():
    """Probar que el modelo responde con datos de entrada definidos"""
    # Descargar datos de prueba
    test_data = download_test_data()
    
    # Obtener la primera imagen de prueba
    test_image = test_data['images'][0]
    image_bytes = base64.b64decode(test_image['image'])
    
    # Preprocesar imagen
    input_data = preprocess_image(image_bytes)
    
    # Cargar modelo
    model_path = "/tmp/densenet121_Opset17.onnx"
    assert os.path.exists(model_path), "El modelo no existe en /tmp/densenet121_Opset17.onnx"
    
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Obtener nombres de entrada/salida
    input_name = session.get_inputs()[0].name
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
    # Descargar datos de prueba
    test_data = download_test_data()
    
    # Cargar modelo
    model_path = "/tmp/densenet121_Opset17.onnx"
    assert os.path.exists(model_path), "El modelo no existe en /tmp/densenet121_Opset17.onnx"
    
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Obtener nombres de entrada/salida
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    correct_predictions = 0
    total_confidence = 0
    
    # Evaluar cada imagen
    for test_case in test_data['images']:
        # Preprocesar imagen
        image_bytes = base64.b64decode(test_case['image'])
        input_data = preprocess_image(image_bytes)
        
        # Ejecutar inferencia
        outputs = session.run([output_name], {input_name: input_data})
        predictions = outputs[0]
        
        # Calcular softmax
        exp_preds = np.exp(predictions - np.max(predictions))
        probabilities = exp_preds / np.sum(exp_preds)
        
        # Obtener clase predicha y confianza
        predicted_class = np.argmax(probabilities)
        confidence = np.max(probabilities)
        
        # Actualizar métricas
        if predicted_class == test_case['label']:
            correct_predictions += 1
        total_confidence += confidence
    
    # Calcular métricas finales
    accuracy = correct_predictions / len(test_data['images'])
    avg_confidence = total_confidence / len(test_data['images'])
    
    # Verificar umbrales
    assert accuracy >= METRICS_THRESHOLD['accuracy'], f"Accuracy {accuracy} below threshold {METRICS_THRESHOLD['accuracy']}"
    assert avg_confidence >= METRICS_THRESHOLD['avg_confidence'], f"Average confidence {avg_confidence} below threshold {METRICS_THRESHOLD['avg_confidence']}" 