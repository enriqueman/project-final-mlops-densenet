import os
import base64
import docker
import boto3
import tarfile
import tempfile
import logging
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO)
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
    """Descargar modelo desde ECR"""
    try:
        # Variables de entorno
        aws_region = os.environ.get('AWS_REGION', 'us-east-1')
        stage = os.environ.get('STAGE', 'dev')
        account_id = os.environ.get('AWS_ACCOUNT_ID') or get_account_id()
        model_repo = f'densenet121-model-{stage}'
        
        if not account_id:
            raise Exception("No se pudo obtener el Account ID")
        
        logger.info(f"Configuración: Region={aws_region}, Stage={stage}, Account={account_id}")
        
        # Configurar ECR
        ecr_client = boto3.client('ecr', region_name=aws_region)
        
        # URI del modelo
        model_uri = f"{account_id}.dkr.ecr.{aws_region}.amazonaws.com/{model_repo}:latest"
        logger.info(f"Descargando modelo desde: {model_uri}")
        
        try:
            # Obtener token de autorización
            token = ecr_client.get_authorization_token()
            username, password = base64.b64decode(
                token['authorizationData'][0]['authorizationToken']
            ).decode().split(':')
            registry = token['authorizationData'][0]['proxyEndpoint']
            
            logger.info("Token de autorización obtenido exitosamente")
        except Exception as e:
            logger.error(f"Error obteniendo token de ECR: {str(e)}")
            raise
        
        try:
            # Cliente Docker
            docker_client = docker.from_env()
            
            # Login a ECR
            docker_client.login(
                username=username,
                password=password,
                registry=registry
            )
            logger.info("Login a ECR exitoso")
        except Exception as e:
            logger.error(f"Error en login a Docker/ECR: {str(e)}")
            raise
        
        try:
            # Pull de la imagen
            logger.info(f"Descargando imagen: {model_uri}")
            image = docker_client.images.pull(model_uri)
            logger.info("Imagen descargada exitosamente")
        except Exception as e:
            logger.error(f"Error descargando imagen: {str(e)}")
            raise
        
        try:
            # Crear contenedor temporal
            logger.info("Creando contenedor temporal")
            container = docker_client.containers.create(model_uri)
            
            # Extraer modelo
            logger.info("Extrayendo modelo del contenedor")
            bits, _ = container.get_archive('/model/densenet121_Opset17.onnx')
            
            # Guardar temporalmente
            with tempfile.NamedTemporaryFile() as tmp:
                for chunk in bits:
                    tmp.write(chunk)
                tmp.flush()
                
                # Extraer archivo
                with tarfile.open(tmp.name) as tar:
                    tar.extractall('/tmp')
            
            logger.info("Modelo extraído exitosamente a /tmp/densenet121_Opset17.onnx")
            
            # Verificar que el archivo existe y tiene tamaño
            model_path = '/tmp/densenet121_Opset17.onnx'
            if not os.path.exists(model_path):
                raise Exception(f"El archivo {model_path} no existe después de la extracción")
            
            size = os.path.getsize(model_path)
            logger.info(f"Tamaño del modelo: {size/1024/1024:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error extrayendo modelo: {str(e)}")
            raise
            
        finally:
            # Limpiar
            try:
                container.remove()
                logger.info("Contenedor temporal eliminado")
            except Exception as e:
                logger.warning(f"Error eliminando contenedor temporal: {str(e)}")
            
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