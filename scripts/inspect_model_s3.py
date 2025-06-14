#!/usr/bin/env python3
"""
Script para inspeccionar modelo DenseNet121 desde S3
"""

import os
import boto3
import logging
import sys
import time
import tempfile
import onnxruntime as ort
from pathlib import Path

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

def inspect_s3_bucket(s3_client, bucket_name):
    """Inspeccionar contenido del bucket S3"""
    try:
        logger.info(f"\n=== Inspección del bucket S3: {bucket_name} ===")
        
        # Verificar que el bucket existe
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"✅ Bucket existe: {bucket_name}")
        except s3_client.exceptions.NoSuchBucket:
            logger.error(f"❌ Bucket no existe: {bucket_name}")
            return False
        except Exception as e:
            logger.error(f"❌ Error accediendo bucket: {str(e)}")
            return False
        
        # Obtener información del bucket
        try:
            bucket_location = s3_client.get_bucket_location(Bucket=bucket_name)
            logger.info(f"📍 Región del bucket: {bucket_location.get('LocationConstraint', 'us-east-1')}")
        except Exception as e:
            logger.warning(f"No se pudo obtener ubicación del bucket: {str(e)}")
        
        # Listar objetos en el bucket
        logger.info(f"\n📁 Contenido del bucket:")
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            total_objects = 0
            total_size = 0
            
            for page in paginator.paginate(Bucket=bucket_name):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_objects += 1
                        total_size += obj['Size']
                        size_mb = obj['Size'] / 1024 / 1024
                        logger.info(f"  📄 {obj['Key']} ({size_mb:.2f} MB) - {obj['LastModified']}")
            
            logger.info(f"\n📊 Resumen del bucket:")
            logger.info(f"  Total objetos: {total_objects}")
            logger.info(f"  Tamaño total: {total_size / 1024 / 1024:.2f} MB")
            
            if total_objects == 0:
                logger.warning("⚠️  El bucket está vacío")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error listando objetos del bucket: {str(e)}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error inspeccionando bucket S3: {str(e)}")
        return False

def inspect_model_object(s3_client, bucket_name, model_key):
    """Inspeccionar el objeto del modelo específicamente"""
    try:
        logger.info(f"\n=== Inspección del modelo: s3://{bucket_name}/{model_key} ===")
        
        # Verificar que el objeto existe
        try:
            obj_info = s3_client.head_object(Bucket=bucket_name, Key=model_key)
            logger.info(f"✅ Modelo encontrado: {model_key}")
        except s3_client.exceptions.NoSuchKey:
            logger.error(f"❌ Modelo no encontrado: {model_key}")
            
            # Buscar archivos similares
            logger.info("🔍 Buscando archivos .onnx en el bucket:")
            try:
                response = s3_client.list_objects_v2(Bucket=bucket_name)
                if 'Contents' in response:
                    onnx_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.onnx')]
                    if onnx_files:
                        logger.info("   Archivos .onnx encontrados:")
                        for file in onnx_files:
                            logger.info(f"     - {file}")
                    else:
                        logger.info("   No se encontraron archivos .onnx")
                else:
                    logger.info("   El bucket está vacío")
            except Exception as search_error:
                logger.warning(f"   Error buscando archivos: {str(search_error)}")
            
            return False
        except Exception as e:
            logger.error(f"❌ Error verificando modelo: {str(e)}")
            return False
        
        # Mostrar información del objeto
        logger.info(f"📊 Información del modelo:")
        logger.info(f"  Tamaño: {obj_info['ContentLength'] / 1024 / 1024:.2f} MB")
        logger.info(f"  Última modificación: {obj_info['LastModified']}")
        logger.info(f"  ETag: {obj_info['ETag']}")
        logger.info(f"  Tipo de contenido: {obj_info.get('ContentType', 'N/A')}")
        
        if 'Metadata' in obj_info and obj_info['Metadata']:
            logger.info(f"  Metadatos:")
            for key, value in obj_info['Metadata'].items():
                logger.info(f"    {key}: {value}")
        
        # Verificar versionado si está habilitado
        try:
            versions = s3_client.list_object_versions(Bucket=bucket_name, Prefix=model_key)
            if 'Versions' in versions:
                version_count = len(versions['Versions'])
                logger.info(f"  Versiones disponibles: {version_count}")
                if version_count > 1:
                    logger.info("  Versiones:")
                    for version in versions['Versions'][:5]:  # Mostrar solo las primeras 5
                        is_current = " (actual)" if version['IsLatest'] else ""
                        logger.info(f"    - {version['VersionId'][:8]}... ({version['LastModified']}){is_current}")
        except Exception as e:
            logger.info(f"  Versionado: No disponible ({str(e)})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error inspeccionando modelo: {str(e)}")
        return False

def validate_model_file(s3_client, bucket_name, model_key):
    """Descargar y validar el archivo del modelo"""
    try:
        logger.info(f"\n=== Validación del modelo ONNX ===")
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Descargar modelo
            logger.info(f"⬇️  Descargando modelo para validación...")
            s3_client.download_file(bucket_name, model_key, tmp_path)
            
            # Verificar tamaño del archivo descargado
            file_size = os.path.getsize(tmp_path)
            logger.info(f"📁 Archivo descargado: {file_size / 1024 / 1024:.2f} MB")
            
            # Validar archivo ONNX
            logger.info(f"🔍 Validando archivo ONNX...")
            
            try:
                # Cargar con ONNX Runtime
                session = ort.InferenceSession(tmp_path, providers=['CPUExecutionProvider'])
                
                # Obtener información del modelo
                logger.info(f"✅ Modelo ONNX válido")
                
                # Inputs
                logger.info(f"📥 Entradas del modelo:")
                for input_meta in session.get_inputs():
                    logger.info(f"  - {input_meta.name}: {input_meta.type} {input_meta.shape}")
                
                # Outputs
                logger.info(f"📤 Salidas del modelo:")
                for output_meta in session.get_outputs():
                    logger.info(f"  - {output_meta.name}: {output_meta.type} {output_meta.shape}")
                
                # Metadatos del modelo
                try:
                    metadata = session.get_modelmeta()
                    if metadata.custom_metadata_map:
                        logger.info(f"🏷️  Metadatos del modelo:")
                        for key, value in metadata.custom_metadata_map.items():
                            logger.info(f"  {key}: {value}")
                    
                    logger.info(f"📋 Información adicional:")
                    logger.info(f"  Descripción: {metadata.description or 'N/A'}")
                    logger.info(f"  Dominio: {metadata.domain or 'N/A'}")
                    logger.info(f"  Versión del modelo: {metadata.version or 'N/A'}")
                    logger.info(f"  Productor: {metadata.producer_name or 'N/A'}")
                    
                except Exception as meta_error:
                    logger.warning(f"No se pudieron obtener metadatos: {str(meta_error)}")
                
                return True
                
            except Exception as onnx_error:
                logger.error(f"❌ Error validando ONNX: {str(onnx_error)}")
                return False
                
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(tmp_path)
            except Exception as cleanup_error:
                logger.warning(f"Error limpiando archivo temporal: {str(cleanup_error)}")
        
    except Exception as e:
        logger.error(f"❌ Error validando modelo: {str(e)}")
        return False

def test_model_download():
    """Probar descarga del modelo usando el script oficial"""
    try:
        logger.info(f"\n=== Prueba de descarga del modelo ===")
        
        # Importar el script de descarga
        try:
            from download_model_from_s3 import download_model_from_s3
            logger.info(f"📦 Script de descarga importado correctamente")
        except ImportError as e:
            logger.error(f"❌ Error importando script de descarga: {str(e)}")
            return False
        
        # Ejecutar descarga
        logger.info(f"⬇️  Ejecutando descarga de prueba...")
        if download_model_from_s3():
            logger.info(f"✅ Descarga de prueba exitosa")
            
            # Verificar que el archivo existe
            possible_paths = [
                '/tmp/densenet121_Opset17.onnx',
                '/code/app/model/densenet121_Opset17.onnx',
                './model/densenet121_Opset17.onnx'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    size = os.path.getsize(path)
                    logger.info(f"📁 Modelo descargado en: {path} ({size / 1024 / 1024:.2f} MB)")
                    return True
            
            logger.warning(f"⚠️  Descarga reportada como exitosa pero no se encontró el archivo")
            return False
        else:
            logger.error(f"❌ Error en descarga de prueba")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error probando descarga: {str(e)}")
        return False

def inspect_model_s3():
    """Función principal de inspección"""
    try:
        # Obtener configuración
        config = get_s3_config()
        bucket_name = config['bucket_name']
        model_key = config['model_key']
        aws_region = config['aws_region']
        stage = config['stage']
        
        logger.info(f"🚀 Iniciando inspección del modelo en S3")
        logger.info(f"📊 Configuración:")
        logger.info(f"  Stage: {stage}")
        logger.info(f"  Región: {aws_region}")
        logger.info(f"  Bucket: {bucket_name}")
        logger.info(f"  Modelo: {model_key}")
        
        # Crear cliente S3
        try:
            s3_client = boto3.client('s3', region_name=aws_region)
            logger.info(f"✅ Cliente S3 creado correctamente")
        except Exception as e:
            logger.error(f"❌ Error creando cliente S3: {str(e)}")
            return False
        
        # Verificar credenciales
        try:
            identity = boto3.client('sts').get_caller_identity()
            logger.info(f"🔑 Credenciales válidas - Account: {identity['Account']}")
        except Exception as e:
            logger.error(f"❌ Error verificando credenciales: {str(e)}")
            return False
        
        success = True
        
        # 1. Inspeccionar bucket
        if not inspect_s3_bucket(s3_client, bucket_name):
            success = False
        
        # 2. Inspeccionar modelo específico
        if not inspect_model_object(s3_client, bucket_name, model_key):
            success = False
        
        # 3. Validar archivo del modelo (solo si existe)
        try:
            s3_client.head_object(Bucket=bucket_name, Key=model_key)
            if not validate_model_file(s3_client, bucket_name, model_key):
                success = False
        except:
            logger.warning(f"⚠️  Saltando validación - modelo no encontrado")
        
        # 4. Probar descarga usando script oficial
        if not test_model_download():
            success = False
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error en inspección principal: {str(e)}")
        return False

def main():
    """Función principal"""
    try:
        logger.info("🔍 Script de Inspección de Modelo S3 - DenseNet121")
        logger.info("=" * 60)
        
        if inspect_model_s3():
            logger.info("\n" + "=" * 60)
            logger.info("✅ Inspección completada exitosamente")
            logger.info("🎉 El modelo está correctamente configurado en S3")
            sys.exit(0)
        else:
            logger.error("\n" + "=" * 60)
            logger.error("❌ Inspección falló")
            logger.error("🔧 Revisa la configuración del bucket y modelo")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error en el proceso principal: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()