import os
import base64
import docker
import boto3
import logging
import sys

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

def inspect_container_contents(container, paths_to_check=['/model', '/models', '/app', '/opt', '/usr/local']):
    """Inspeccionar contenidos del contenedor"""
    logger.info("\n=== Inspección del contenedor ===")
    
    # Buscar archivos .onnx
    logger.info("\nBuscando archivos .onnx:")
    exec_result = container.exec_run('find / -name "*.onnx" 2>/dev/null')
    onnx_files = exec_result.output.decode().strip().split('\n')
    for file in onnx_files:
        if file:
            logger.info(f"Encontrado: {file}")
    
    # Inspeccionar directorios comunes
    logger.info("\nInspeccionando directorios comunes:")
    for path in paths_to_check:
        logger.info(f"\nContenido de {path}:")
        exec_result = container.exec_run(f'ls -la {path} 2>/dev/null')
        output = exec_result.output.decode().strip()
        if output:
            logger.info(output)
        else:
            logger.info(f"No se pudo acceder a {path}")
    
    # Verificar variables de entorno
    logger.info("\nVariables de entorno:")
    exec_result = container.exec_run('env')
    logger.info(exec_result.output.decode())
    
    # Verificar sistema de archivos
    logger.info("\nEstructura del sistema de archivos:")
    exec_result = container.exec_run('df -h')
    logger.info(exec_result.output.decode())

def inspect_model_image():
    """Inspeccionar imagen del modelo"""
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
        logger.info(f"Inspeccionando imagen: {model_uri}")
        
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
            
            # Inspeccionar imagen
            logger.info("\n=== Información de la imagen ===")
            image_info = docker_client.api.inspect_image(image.id)
            logger.info(f"ID: {image.id}")
            logger.info(f"Tags: {image.tags}")
            logger.info(f"Creada: {image_info['Created']}")
            logger.info(f"Tamaño: {image_info['Size']/1024/1024:.2f} MB")
            logger.info(f"Entrypoint: {image_info['Config']['Entrypoint']}")
            logger.info(f"Cmd: {image_info['Config']['Cmd']}")
            logger.info(f"WorkingDir: {image_info['Config']['WorkingDir']}")
            
            # Crear contenedor temporal para inspección
            logger.info("\nCreando contenedor temporal para inspección...")
            container = docker_client.containers.create(
                model_uri,
                command='sleep 1000',  # Mantener el contenedor vivo
                tty=True
            )
            container.start()
            
            # Inspeccionar contenidos
            inspect_container_contents(container)
            
            return True
            
        except Exception as e:
            logger.error(f"Error inspeccionando imagen: {str(e)}")
            raise
            
        finally:
            # Limpiar
            try:
                container.stop()
                container.remove()
                logger.info("\nContenedor temporal eliminado")
            except Exception as e:
                logger.warning(f"Error eliminando contenedor temporal: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error en la inspección: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        if not inspect_model_image():
            logger.error("Fallo en la inspección de la imagen")
            sys.exit(1)
        logger.info("Proceso completado exitosamente")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}")
        sys.exit(1) 