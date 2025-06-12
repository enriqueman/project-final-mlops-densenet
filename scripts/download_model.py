import os
import base64
import boto3
import logging
import sys
import tempfile
import subprocess

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_account_id():
    """Obtener Account ID de AWS"""
    try:
        sts_client = boto3.client('sts')
        return sts_client.get_caller_identity()['Account']
    except Exception as e:
        logger.error(f"Error obteniendo Account ID: {str(e)}")
        return None

def download_model_from_ecr():
    """Descargar modelo desde ECR usando docker CLI"""
    try:
        # Variables de entorno
        aws_region = os.environ.get('AWS_REGION', 'us-east-1')
        stage = os.environ.get('STAGE', 'dev')
        account_id = os.environ.get('AWS_ACCOUNT_ID') or get_account_id()
        model_repo = os.environ.get('MODEL_REPOSITORY', f'densenet121-model-{stage}')
        
        if not account_id:
            raise Exception("No se pudo obtener el Account ID")
        
        logger.info(f"Configuración: Region={aws_region}, Stage={stage}, Account={account_id}")
        logger.info(f"Repositorio del modelo: {model_repo}")
        
        # URI del modelo
        model_uri = f"{account_id}.dkr.ecr.{aws_region}.amazonaws.com/{model_repo}:latest"
        logger.info(f"Descargando modelo desde: {model_uri}")
        
        # Login a ECR usando AWS CLI
        logger.info("Autenticando con ECR...")
        result = subprocess.run([
            'aws', 'ecr', 'get-login-password', '--region', aws_region
        ], capture_output=True, text=True, check=True)
        
        login_password = result.stdout.strip()
        
        # Login a Docker usando el password
        login_cmd = [
            'docker', 'login', '--username', 'AWS', '--password-stdin',
            f"{account_id}.dkr.ecr.{aws_region}.amazonaws.com"
        ]
        
        result = subprocess.run(
            login_cmd,
            input=login_password,
            text=True,
            check=True,
            capture_output=True
        )
        logger.info("Login a ECR exitoso")
        
        # Crear contenedor temporal para extraer modelo
        logger.info("Creando contenedor temporal...")
        create_cmd = ['docker', 'create', model_uri]
        result = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
        container_id = result.stdout.strip()
        
        try:
            # Extraer modelo del contenedor
            logger.info("Extrayendo modelo...")
            copy_cmd = [
                'docker', 'cp', 
                f"{container_id}:/opt/ml/model/densenet121_Opset17.onnx",
                "/code/app/model/densenet121_Opset17.onnx"
            ]
            subprocess.run(copy_cmd, check=True)
            
            # Verificar que el modelo se extrajo correctamente
            model_path = "/code/app/model/densenet121_Opset17.onnx"
            if not os.path.exists(model_path):
                raise Exception(f"El archivo {model_path} no existe después de la extracción")
            
            size = os.path.getsize(model_path)
            logger.info(f"Modelo extraído exitosamente ({size/1024/1024:.2f} MB)")
            
            return True
            
        finally:
            # Limpiar contenedor temporal
            try:
                subprocess.run(['docker', 'rm', container_id], check=True)
                logger.info("Contenedor temporal eliminado")
            except Exception as e:
                logger.warning(f"Error eliminando contenedor temporal: {str(e)}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error en comando: {e.cmd}")
        logger.error(f"Código de salida: {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error descargando modelo: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        if not download_model_from_ecr():
            logger.error("Fallo en la descarga del modelo")
            sys.exit(1)
        logger.info("Proceso completado exitosamente")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}")
        sys.exit(1)