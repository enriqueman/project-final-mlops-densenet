import os
import json
import base64
import logging
import boto3
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import sys
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URLs de imágenes de ejemplo para pruebas (ImageNet)
TEST_IMAGES = [
    {
        "url": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        "label": 258,  # Perro (ImageNet)
        "filename": "dog.jpg"
    },
    {
        "url": "https://raw.githubusercontent.com/pytorch/hub/master/images/cat.jpg",
        "label": 281,  # Gato (ImageNet)
        "filename": "cat.jpg"
    }
]

def download_and_process_image(url):
    """Descargar y procesar imagen"""
    try:
        logger.info(f"Descargando imagen desde: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Guardar imagen original
        img_bytes_original = BytesIO()
        img.save(img_bytes_original, format="JPEG")
        img_bytes_original.seek(0)
        
        # Redimensionar a 224x224 para el modelo
        img_resized = img.resize((224, 224))
        
        # Guardar imagen redimensionada
        img_bytes_resized = BytesIO()
        img_resized.save(img_bytes_resized, format="JPEG")
        img_bytes_resized.seek(0)
        
        logger.info(f"Imagen procesada exitosamente: {url}")
        return {
            'original': img_bytes_original.getvalue(),
            'resized': img_bytes_resized.getvalue(),
            'base64': base64.b64encode(img_bytes_resized.getvalue()).decode()
        }
    except Exception as e:
        logger.error(f"Error procesando imagen {url}: {str(e)}")
        raise

def generate_test_data():
    """Generar datos de prueba"""
    test_data = {
        "images": [],
        "metadata": {
            "model": "densenet121",
            "input_shape": [1, 3, 224, 224],
            "preprocessing": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize": [224, 224]
            }
        }
    }
    
    processed_images = []
    
    # Procesar cada imagen
    for img_info in TEST_IMAGES:
        try:
            img_data = download_and_process_image(img_info["url"])
            
            # Agregar al conjunto de datos
            test_data["images"].append({
                "image": img_data['base64'],
                "label": img_info["label"],
                "filename": img_info["filename"]
            })
            
            # Guardar para subir a S3
            processed_images.append({
                "original": img_data['original'],
                "resized": img_data['resized'],
                "filename": img_info["filename"]
            })
            
            logger.info(f"Imagen agregada al conjunto de datos: {img_info['filename']}")
        except Exception as e:
            logger.error(f"Error procesando imagen {img_info['url']}: {str(e)}")
            continue
    
    if not test_data["images"]:
        raise Exception("No se pudo procesar ninguna imagen")
    
    return test_data, processed_images

def wait_for_bucket(bucket_name, max_attempts=5):
    """Esperar a que el bucket esté disponible"""
    s3 = boto3.client('s3')
    for attempt in range(max_attempts):
        try:
            s3.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} está disponible")
            return True
        except Exception:
            if attempt == max_attempts - 1:
                logger.error(f"Bucket {bucket_name} no está disponible después de {max_attempts} intentos")
                return False
            logger.warning(f"Bucket {bucket_name} no está disponible, reintentando... ({attempt + 1}/{max_attempts})")
            time.sleep(5)
    return False

def upload_to_s3(bucket, test_data, processed_images):
    """Subir datos a S3"""
    if not wait_for_bucket(bucket):
        raise Exception(f"El bucket {bucket} no está disponible")

    s3 = boto3.client('s3')
    try:
        # Subir JSON con datos de prueba
        logger.info("Subiendo archivo JSON con datos de prueba...")
        s3.put_object(
            Bucket=bucket,
            Key='test_data/test_images.json',
            Body=json.dumps(test_data),
            ContentType='application/json'
        )
        
        # Subir imágenes originales y redimensionadas
        for img in processed_images:
            # Subir imagen original
            logger.info(f"Subiendo imagen original: {img['filename']}")
            s3.put_object(
                Bucket=bucket,
                Key=f'test_data/images/original/{img["filename"]}',
                Body=img['original'],
                ContentType='image/jpeg'
            )
            
            # Subir imagen redimensionada
            logger.info(f"Subiendo imagen redimensionada: {img['filename']}")
            s3.put_object(
                Bucket=bucket,
                Key=f'test_data/images/resized/{img["filename"]}',
                Body=img['resized'],
                ContentType='image/jpeg'
            )
        
        logger.info(f"Todos los datos subidos exitosamente a s3://{bucket}/")
        return True
    except Exception as e:
        logger.error(f"Error subiendo datos a S3: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Obtener variables de entorno
        stage = os.environ.get('STAGE', 'dev')
        bucket_name = f'densenet-test-data-{stage}'
        
        # Generar datos
        logger.info("Generando datos de prueba...")
        test_data, processed_images = generate_test_data()
        
        # Subir a S3
        logger.info(f"Subiendo datos a bucket {bucket_name}...")
        upload_to_s3(bucket_name, test_data, processed_images)
        
        logger.info("Proceso completado exitosamente")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error en el proceso: {str(e)}")
        sys.exit(1) 