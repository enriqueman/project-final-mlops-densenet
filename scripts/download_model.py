import os
import base64
import docker
import boto3
import tarfile
import tempfile
import logging

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
        
        # Configurar ECR
        ecr_client = boto3.client('ecr', region_name=aws_region)
        
        # URI del modelo
        model_uri = f"{account_id}.dkr.ecr.{aws_region}.amazonaws.com/{model_repo}:latest"
        logger.info(f"Descargando modelo desde: {model_uri}")
        
        # Obtener token de autorización
        token = ecr_client.get_authorization_token()
        username, password = base64.b64decode(
            token['authorizationData'][0]['authorizationToken']
        ).decode().split(':')
        registry = token['authorizationData'][0]['proxyEndpoint']
        
        # Cliente Docker
        docker_client = docker.from_env()
        
        # Login a ECR
        docker_client.login(
            username=username,
            password=password,
            registry=registry
        )
        
        # Pull de la imagen
        logger.info("Descargando imagen...")
        image = docker_client.images.pull(model_uri)
        
        # Crear contenedor temporal
        container = docker_client.containers.create(model_uri)
        
        try:
            # Extraer modelo
            logger.info("Extrayendo modelo...")
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
            return True
            
        finally:
            # Limpiar
            container.remove()
            
    except Exception as e:
        logger.error(f"Error descargando modelo: {str(e)}")
        return False

if __name__ == "__main__":
    download_model_from_ecr() 