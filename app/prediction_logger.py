import boto3
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import tempfile
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class PredictionLogger:
    """Servicio para gestionar logs de predicciones en S3"""
    
    def __init__(self):
        self.stage = os.getenv('STAGE', 'dev')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.bucket_name = "predics"
        self.folder_name = f"logs{self.stage}"  # logsdev o logsprod
        self.file_name = f"predicciones_{self.stage}.txt"
        self.file_key = f"{self.folder_name}/{self.file_name}"
        
        # Inicializar cliente S3
        try:
            self.s3_client = boto3.client('s3', region_name=self.aws_region)
            logger.info(f"Cliente S3 inicializado para bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Error inicializando cliente S3: {str(e)}")
            self.s3_client = None
    
    def _ensure_bucket_exists(self) -> bool:
        """Asegurar que el bucket existe"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    # Crear bucket
                    if self.aws_region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.aws_region}
                        )
                    logger.info(f"Bucket {self.bucket_name} creado exitosamente")
                    return True
                except Exception as create_error:
                    logger.error(f"Error creando bucket: {str(create_error)}")
                    return False
            else:
                logger.error(f"Error accediendo bucket: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Error verificando bucket: {str(e)}")
            return False
    
    def _get_existing_content(self) -> str:
        """Obtener contenido existente del archivo de logs"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.file_key)
            return response['Body'].read().decode('utf-8')
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                # Archivo no existe, crear header
                header = f"# Predicciones para {self.stage.upper()} - Iniciado el {datetime.now().isoformat()}\n"
                header += "# Formato: timestamp|filename|top_prediction|confidence|processing_time|all_predictions\n\n"
                return header
            else:
                logger.error(f"Error obteniendo archivo existente: {str(e)}")
                return ""
        except Exception as e:
            logger.error(f"Error inesperado obteniendo archivo: {str(e)}")
            return ""
    
    def log_prediction(self, 
                      filename: str,
                      predictions: list,
                      confidence_scores: list,
                      processing_time: float) -> bool:
        """Registrar una nueva predicción en el archivo de logs"""
        
        if not self.s3_client:
            logger.error("Cliente S3 no disponible")
            return False
        
        try:
            # Verificar que el bucket existe
            if not self._ensure_bucket_exists():
                logger.error("No se pudo crear/acceder al bucket")
                return False
            
            # Obtener contenido existente
            existing_content = self._get_existing_content()
            
            # Preparar nueva línea
            timestamp = datetime.now().isoformat()
            top_prediction = predictions[0] if predictions else "unknown"
            top_confidence = confidence_scores[0] if confidence_scores else 0.0
            
            # Crear línea de log
            all_predictions_str = json.dumps({
                "predictions": predictions,
                "confidence_scores": confidence_scores
            })
            
            new_line = f"{timestamp}|{filename}|{top_prediction}|{top_confidence:.4f}|{processing_time:.4f}|{all_predictions_str}\n"
            
            # Combinar contenido
            updated_content = existing_content + new_line
            
            # Subir archivo actualizado
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.file_key,
                Body=updated_content.encode('utf-8'),
                ContentType='text/plain'
            )
            
            logger.info(f"Predicción registrada exitosamente en s3://{self.bucket_name}/{self.file_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error registrando predicción: {str(e)}")
            return False
    
    def get_logs_content(self) -> Optional[str]:
        """Obtener contenido completo del archivo de logs"""
        if not self.s3_client:
            return None
        
        try:
            if not self._ensure_bucket_exists():
                return None
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.file_key)
            return response['Body'].read().decode('utf-8')
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                return f"# No hay predicciones registradas para {self.stage.upper()} aún\n"
            else:
                logger.error(f"Error obteniendo logs: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Error inesperado obteniendo logs: {str(e)}")
            return None
    
    def download_logs_file(self) -> Optional[bytes]:
        """Descargar archivo de logs como bytes para descarga"""
        content = self.get_logs_content()
        if content is not None:
            return content.encode('utf-8')
        return None
    
    def get_logs_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del archivo de logs"""
        content = self.get_logs_content()
        if not content:
            return {
                "total_predictions": 0,
                "file_exists": False,
                "last_prediction": None
            }
        
        lines = content.strip().split('\n')
        # Filtrar líneas que no son comentarios
        data_lines = [line for line in lines if line and not line.startswith('#')]
        
        stats = {
            "total_predictions": len(data_lines),
            "file_exists": True,
            "file_size_bytes": len(content.encode('utf-8')),
            "last_prediction": None,
            "stage": self.stage,
            "bucket": self.bucket_name,
            "file_key": self.file_key
        }
        
        if data_lines:
            # Obtener última predicción
            last_line = data_lines[-1]
            try:
                parts = last_line.split('|')
                if len(parts) >= 4:
                    stats["last_prediction"] = {
                        "timestamp": parts[0],
                        "filename": parts[1],
                        "top_prediction": parts[2],
                        "confidence": float(parts[3])
                    }
            except Exception as e:
                logger.warning(f"Error parseando última predicción: {str(e)}")
        
        return stats