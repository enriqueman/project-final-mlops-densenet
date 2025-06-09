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

def find_model_path(container):
    """Encontrar la ruta del modelo en el contenedor"""
    try:
        # Buscar archivos .onnx
        exec_result = container.exec_run('find / -name "*.onnx" 2>/dev/null')
        paths = exec_result.output.decode().strip().split('\n')
        
        # Filtrar rutas válidas
        valid_paths = [p for p in paths if p and 'densenet' in p.lower()]
        
        if valid_paths:
            logger.info(f"Modelo encontrado en: {valid_paths[0]}")
            return valid_paths[0]
        
        # Si no encontramos el archivo, listar directorios comunes
        common_dirs = ['/model', '/models', '/app', '/opt', '/usr/local']
        for dir in common_dirs:
            exec_result = container.exec_run(f'ls -la {dir}')
            logger.info(f"Contenido de {dir}:")
            logger.info(exec_result.output.decode())
        
        raise Exception("No se encontró el archivo del modelo")
    except Exception as e:
        logger.error(f"Error buscando modelo: {str(e)}")
        raise

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
            container = docker_client.containers.create(
                model_uri,
                command='sleep 1000',  # Mantener el contenedor vivo
                tty=True
            )
            container.start()
            
            # Encontrar la ruta del modelo
            model_path = find_model_path(container)
            
            # Extraer modelo
            logger.info(f"Extrayendo modelo desde: {model_path}")
            bits, _ = container.get_archive(model_path)
            
            # Guardar temporalmente
            with tempfile.NamedTemporaryFile() as tmp:
                for chunk in bits:
                    tmp.write(chunk)
                tmp.flush()
                
                # Extraer archivo
                with tarfile.open(tmp.name) as tar:
                    # Obtener el nombre del archivo dentro del tar
                    members = tar.getmembers()
                    if not members:
                        raise Exception("Archivo tar vacío")
                    
                    # Renombrar el archivo al extraer
                    member = members[0]
                    member.name = "densenet121_Opset17.onnx"
                    
                    # Extraer a /tmp
                    tar.extract(member, "/tmp")
            
            # Verificar que el archivo existe y tiene tamaño
            output_path = '/tmp/densenet121_Opset17.onnx'
            if not os.path.exists(output_path):
                raise Exception(f"El archivo {output_path} no existe después de la extracción")
            
            size = os.path.getsize(output_path)
            logger.info(f"Modelo extraído exitosamente a {output_path} ({size/1024/1024:.2f} MB)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error extrayendo modelo: {str(e)}")
            raise
            
        finally:
            # Limpiar
            try:
                container.stop()
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