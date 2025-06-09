import os
import json
import base64
import boto3
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# URLs de imágenes de ejemplo para pruebas (ImageNet)
TEST_IMAGES = [
    {
        "url": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        "label": 258  # Perro (ImageNet)
    },
    {
        "url": "https://raw.githubusercontent.com/pytorch/hub/master/images/cat.jpg",
        "label": 281  # Gato (ImageNet)
    }
]

def download_and_encode_image(url):
    """Descargar imagen y codificar en base64"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    # Convertir a RGB si es necesario
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionar a 224x224
    img = img.resize((224, 224))
    
    # Convertir a base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str

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
    
    # Procesar cada imagen
    for img_info in TEST_IMAGES:
        try:
            img_str = download_and_encode_image(img_info["url"])
            test_data["images"].append({
                "image": img_str,
                "label": img_info["label"]
            })
        except Exception as e:
            print(f"Error procesando imagen {img_info['url']}: {str(e)}")
    
    return test_data

def upload_to_s3(data, bucket, key):
    """Subir datos a S3"""
    s3 = boto3.client('s3')
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data),
            ContentType='application/json'
        )
        print(f"Datos subidos exitosamente a s3://{bucket}/{key}")
        return True
    except Exception as e:
        print(f"Error subiendo datos a S3: {str(e)}")
        return False

if __name__ == "__main__":
    # Obtener variables de entorno
    stage = os.environ.get('STAGE', 'dev')
    bucket_name = f'densenet-test-data-{stage}'
    
    # Generar datos
    print("Generando datos de prueba...")
    test_data = generate_test_data()
    
    # Subir a S3
    print(f"Subiendo datos a bucket {bucket_name}...")
    upload_to_s3(test_data, bucket_name, 'test_data/test_images.json') 