import os
import base64
import docker
import boto3
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

def inspect_container_contents(container):
    """Inspeccionar contenidos del contenedor"""
    try:
        logger.info("\n=== Inspección del contenedor del modelo ===")
        
        # Verificar que el contenedor está corriendo
        container.reload()
        if container.status != 'running':
            logger.error(f"El contenedor no está corriendo (estado: {container.status})")
            return False
        
        # Buscar archivos .onnx
        logger.info("\nBuscando archivos .onnx:")
        exec_result = container.exec_run('find / -name "*.onnx" 2>/dev/null')
        if exec_result.exit_code == 0:
            onnx_files = exec_result.output.decode().strip().split('\n')
            for file in onnx_files:
                if file:
                    logger.info(f"Encontrado: {file}")
                    # Intentar obtener más información del archivo
                    file_info = container.exec_run(f'ls -l {file}')
                    if file_info.exit_code == 0:
                        logger.info(f"Detalles: {file_info.output.decode().strip()}")
        else:
            logger.warning("No se pudieron buscar archivos .onnx")
        
        # Directorios a inspeccionar
        paths_to_check = [
            '/',           # Raíz
            '/model',      # Directorio común para modelos
            '/models',     # Directorio común para modelos
            '/app',        # Directorio de la aplicación
            '/workspace'   # Directorio de trabajo
        ]
        
        # Inspeccionar directorios comunes
        logger.info("\nInspeccionando directorios comunes:")
        for path in paths_to_check:
            logger.info(f"\nContenido de {path}:")
            # Primero verificar si el directorio existe
            test_dir = container.exec_run(f'test -d {path}')
            if test_dir.exit_code == 0:
                # Listar contenido con detalles
                exec_result = container.exec_run(f'ls -la {path}')
                output = exec_result.output.decode().strip()
                if output:
                    logger.info(output)
                    # Si es un directorio importante, listar también recursivamente
                    if path in ['/', '/model', '/models', '/app']:
                        logger.info(f"\nListado recursivo de {path}:")
                        exec_result = container.exec_run(f'find {path} -type f')
                        logger.info(exec_result.output.decode().strip())
            else:
                logger.info(f"El directorio {path} no existe")
        
        return True
        
    except Exception as e:
        logger.error(f"Error inspeccionando contenedor: {str(e)}")
        return False

def inspect_model_image():
    """Inspeccionar imagen del modelo"""
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
        logger.info(f"Inspeccionando imagen del modelo: {model_uri}")
        
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
            logger.info("\n=== Información de la imagen del modelo ===")
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
                entrypoint=["/bin/sh", "-c", "tail -f /dev/null"],  # Mantener el contenedor vivo
                detach=True,
                tty=True
            )
            
            # Iniciar contenedor
            container.start()
            
            # Esperar un momento para asegurarnos que el contenedor está corriendo
            time.sleep(2)
            
            # Verificar estado del contenedor
            container.reload()
            if container.status != 'running':
                raise Exception(f"El contenedor no se inició correctamente (estado: {container.status})")
            
            # Inspeccionar contenidos
            inspect_container_contents(container)
            
            return True
            
        except Exception as e:
            logger.error(f"Error inspeccionando imagen: {str(e)}")
            raise
            
        finally:
            # Limpiar
            try:
                if 'container' in locals():
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
            logger.error("Fallo en la inspección de la imagen del modelo")
            sys.exit(1)
        logger.info("Proceso completado exitosamente")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}")
        sys.exit(1) 