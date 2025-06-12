import os
import base64
import docker
import boto3
import tarfile
import tempfile
import logging
import sys
import time

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

def find_model_path(container):
    """Encontrar la ruta del modelo en el contenedor"""
    try:
        logger.info("=== Buscando archivo del modelo ===")
        
        # Ruta esperada del modelo basada en las capas de la imagen
        expected_path = '/opt/ml/model/densenet121_Opset17.onnx'
        logger.info(f"Verificando ruta esperada: {expected_path}")
        
        # Verificar si el archivo existe en la ruta esperada
        test_file = container.exec_run(f'test -f {expected_path}')
        if test_file.exit_code == 0:
            logger.info(f"Modelo encontrado en la ruta esperada: {expected_path}")
            # Verificar el tipo de archivo
            file_type = container.exec_run(f'file {expected_path}')
            if file_type.exit_code == 0:
                logger.info(f"Tipo de archivo: {file_type.output.decode().strip()}")
            # Verificar tamaño
            file_size = container.exec_run(f'ls -lh {expected_path}')
            if file_size.exit_code == 0:
                logger.info(f"Detalles del archivo: {file_size.output.decode().strip()}")
            return expected_path
        
        logger.info("Modelo no encontrado en la ruta esperada, buscando en otras ubicaciones...")
        
        # Buscar archivos .onnx
        logger.info("Buscando archivos .onnx:")
        exec_result = container.exec_run('find / -name "*.onnx" 2>/dev/null')
        if exec_result.exit_code == 0:
            onnx_files = exec_result.output.decode().strip().split('\n')
            for file in onnx_files:
                if file:
                    logger.info(f"Encontrado archivo ONNX: {file}")
                    return file
        
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
        model_repo = os.environ.get('MODEL_REPOSITORY', f'densenet121-model-{stage}')
        
        if not account_id:
            raise Exception("No se pudo obtener el Account ID")
        
        logger.info(f"Configuración: Region={aws_region}, Stage={stage}, Account={account_id}")
        logger.info(f"Repositorio del modelo: {model_repo}")
        
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
            logger.info("Creando contenedor temporal...")
            container = docker_client.containers.create(
                model_uri,
                entrypoint=["/bin/sh", "-c", "tail -f /dev/null"],
                detach=True,
                tty=True
            )
            container.start()
            
            # Esperar un momento para asegurarnos que el contenedor está corriendo
            time.sleep(2)
            
            # Verificar estado del contenedor
            container.reload()
            if container.status != 'running':
                raise Exception(f"El contenedor no se inició correctamente (estado: {container.status})")
            
            # Encontrar la ruta del modelo
            model_path = find_model_path(container)
            
            # Extraer modelo
            logger.info("Extrayendo modelo...")
            bits, _ = container.get_archive(model_path)
            
            # Determinar la ruta de salida según el entorno
            # Si estamos en CI/CD (GitHub Actions), usar /tmp
            # Si estamos en contenedor Docker, usar /code/app/model
            if os.path.exists('/github/workspace') or os.environ.get('GITHUB_ACTIONS'):
                # Estamos en GitHub Actions
                output_dir = '/tmp'
                output_path = '/tmp/densenet121_Opset17.onnx'
                logger.info("Detectado entorno de CI/CD, usando /tmp")
            else:
                # Estamos en contenedor Docker
                output_dir = '/code/app/model'
                output_path = '/code/app/model/densenet121_Opset17.onnx'
                logger.info("Detectado entorno de contenedor, usando /code/app/model")
                # Crear directorio si no existe
                os.makedirs(output_dir, exist_ok=True)
            
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
                    
                    # Extraer al directorio correcto
                    tar.extract(member, output_dir)
            
            # Verificar que el archivo existe y tiene tamaño
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
                if 'container' in locals():
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