#!/usr/bin/env python3
"""
Script para descargar modelo DenseNet121 desde S3
"""

import os
import sys
import boto3
import logging
from pathlib import Path
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_s3_config():
    """Obtener configuración de S3 basada en el stage"""
    stage = os.environ.get('STAGE', 'dev')
    aws_region = os.environ.get('AWS_REGION', 'us-east-1')
    
    # Nombre del bucket según el stage
    bucket_name = f"densenet-models-{stage}"
    
    # Clave del modelo en S3
    model_key = "models/densenet121_Opset17.onnx"
    
    return {
        'bucket_name': bucket_name,
        'model_key': model_key,
        'aws_region': aws_region,
        'stage': stage
    }

def determine_output_path():
    """Determinar la ruta de salida según el entorno"""
    # Detectar si estamos en contenedor Docker o en CI/CD
    if os.path.exists('/code/app'):
        # Estamos en contenedor Docker
        output_dir = '/code/app/model'
        output_path = '/code/app/model/densenet121_Opset17.onnx'
        logger.info("Detectado entorno de contenedor Docker")
    elif os.path.exists('/tmp') and (os.environ.get('GITHUB_ACTIONS') or os.path.exists('/github/workspace')):
        # Estamos en GitHub Actions CI/CD
        output_dir = '/tmp'
        output_path = '/tmp/densenet121_Opset17.onnx'
        logger.info("Detectado entorno de CI/CD (GitHub Actions)")
    else:
        # Entorno por defecto
        output_dir = './model'
        output_path = './model/densenet121_Opset17.onnx'
        logger.info("Usando entorno por defecto")
    
    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    return output_path

def wait_for_bucket(s3_client, bucket_name, max_attempts=5):
    """Esperar a que el bucket esté disponible"""
    for attempt in range(max_attempts):
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} está disponible")
            return True
        except s3_client.exceptions.NoSuchBucket:
            logger.warning(f"Bucket {bucket_name} no existe")
            return False
        except Exception as e:
            if attempt == max_attempts - 1:
                logger.error(f"Bucket {bucket_name} no está disponible después de {max_attempts} intentos: {str(e)}")
                return False
            logger.warning(f"Bucket {bucket_name} no está disponible, reintentando... ({attempt + 1}/{max_attempts})")
            time.sleep(3)
    return False

def download_model_from_s3():
    """Descargar modelo desde S3"""
    try:
        # Obtener configuración
        config = get_s3_config()
        bucket_name = config['bucket_name']
        model_key = config['model_key']
        aws_region = config['aws_region']
        stage = config['stage']
        
        logger.info(f"Configuración: Stage={stage}, Region={aws_region}")
        logger.info(f"Bucket: {bucket_name}")
        logger.info(f"Modelo: {model_key}")
        
        # Determinar ruta de salida
        output_path = determine_output_path()
        logger.info(f"Ruta de salida: {output_path}")
        
        # Crear cliente S3
        try:
            s3_client = boto3.client('s3', region_name=aws_region)
            logger.info("Cliente S3 creado exitosamente")
        except Exception as e:
            logger.error(f"Error creando cliente S3: {str(e)}")
            raise
        
        # Verificar que el bucket existe
        if not wait_for_bucket(s3_client, bucket_name):
            raise Exception(f"Bucket {bucket_name} no está disponible")
        
        # Verificar que el objeto existe
        try:
            s3_client.head_object(Bucket=bucket_name, Key=model_key)
            logger.info(f"Modelo encontrado en S3: s3://{bucket_name}/{model_key}")
        except s3_client.exceptions.NoSuchKey:
            logger.error(f"El modelo no existe en S3: s3://{bucket_name}/{model_key}")
            # Listar objetos disponibles para debug
            try:
                response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="models/")
                if 'Contents' in response:
                    logger.info("Objetos disponibles en el bucket:")
                    for obj in response['Contents']:
                        logger.info(f"  - {obj['Key']} ({obj['Size']} bytes)")
                else:
                    logger.info("No hay objetos en el prefix 'models/'")
            except Exception as list_error:
                logger.warning(f"No se pudo listar objetos del bucket: {str(list_error)}")
            raise
        except Exception as e:
            logger.error(f"Error verificando objeto en S3: {str(e)}")
            raise
        
        # Descargar modelo
        try:
            logger.info(f"Descargando modelo desde S3...")
            
            # Obtener información del objeto para mostrar progreso
            obj_info = s3_client.head_object(Bucket=bucket_name, Key=model_key)
            file_size = obj_info['ContentLength']
            logger.info(f"Tamaño del archivo: {file_size / 1024 / 1024:.2f} MB")
            
            # Descargar archivo
            s3_client.download_file(bucket_name, model_key, output_path)
            
            # Verificar descarga
            if not os.path.exists(output_path):
                raise Exception(f"El archivo no existe después de la descarga: {output_path}")
            
            downloaded_size = os.path.getsize(output_path)
            if downloaded_size != file_size:
                raise Exception(f"Tamaño del archivo descargado ({downloaded_size}) no coincide con el esperado ({file_size})")
            
            logger.info(f"✅ Modelo descargado exitosamente: {output_path}")
            logger.info(f"✅ Tamaño verificado: {downloaded_size / 1024 / 1024:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error descargando modelo: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error en download_model_from_s3: {str(e)}")
        return False

def main():
    """Función principal"""
    try:
        logger.info("🚀 Iniciando descarga de modelo desde S3...")
        
        if download_model_from_s3():
            logger.info("✅ Proceso completado exitosamente")
            sys.exit(0)
        else:
            logger.error("❌ Fallo en la descarga del modelo")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error en el proceso principal: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()