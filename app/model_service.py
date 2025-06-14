import onnxruntime as ort
import numpy as np
from PIL import Image
import time
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, model_path: str):
        """Inicializar el servicio del modelo"""
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.class_names = None
        
        self._load_model()
        self._load_class_names()
    
    def _load_model(self):
        """Cargar el modelo ONNX"""
        try:
            # Crear sesión ONNX Runtime
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Obtener información de entrada y salida
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            logger.info(f"Modelo cargado: {self.model_path}")
            logger.info(f"Input shape: {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def _load_class_names(self):
        """Cargar nombres de clases para DenseNet (ImageNet)"""
        # DenseNet121 entrenado en ImageNet tiene 1000 clases
        # Estas son algunas clases comunes de ImageNet en orden
        self.class_names = [
            "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead", 
            "electric_ray", "stingray", "cock", "hen", "ostrich", "brambling", 
            "goldfinch", "house_finch", "junco", "indigo_bunting", "robin", 
            "bulbul", "jay", "magpie", "chickadee", "water_ouzel", "kite", 
            "bald_eagle", "vulture", "great_grey_owl", "European_fire_salamander",
            "common_newt", "eft", "spotted_salamander", "axolotl", "bullfrog",
            "tree_frog", "tailed_frog", "loggerhead", "leatherback_turtle",
            "mud_turtle", "terrapin", "box_turtle", "banded_gecko", "common_iguana",
            "American_chameleon", "whiptail", "agama", "frilled_lizard",
            "alligator_lizard", "Gila_monster", "green_lizard", "African_chameleon",
            "Komodo_dragon", "African_crocodile", "American_alligator", "triceratops",
            "thunder_snake", "ringneck_snake", "hognose_snake", "green_snake",
            "king_snake", "garter_snake", "water_snake", "vine_snake",
            "night_snake", "boa_constrictor", "rock_python", "Indian_cobra",
            "green_mamba", "sea_snake", "horned_viper", "diamondback",
            "sidewinder", "trilobite", "harvestman", "scorpion", "black_and_gold_garden_spider",
            "barn_spider", "garden_spider", "black_widow", "tarantula",
            "wolf_spider", "tick", "centipede", "black_grouse", "ptarmigan",
            "ruffed_grouse", "prairie_chicken", "peacock", "quail", "partridge",
            "African_grey", "macaw", "sulphur-crested_cockatoo", "lorikeet",
            "coucal", "bee_eater", "hornbill", "hummingbird", "jacamar",
            "toucan", "drake", "red-breasted_merganser", "goose", "black_swan",
            "tusker", "echidna", "platypus", "wallaby", "koala", "wombat",
            "jellyfish", "sea_anemone", "brain_coral", "flatworm", "nematode",
            "conch", "snail", "slug", "sea_slug", "chiton", "chambered_nautilus",
            "Dungeness_crab", "rock_crab", "fiddler_crab", "king_crab", "American_lobster",
            "spiny_lobster", "crayfish", "hermit_crab", "isopod", "white_stork",
            "black_stork", "spoonbill", "flamingo", "little_blue_heron",
            "American_egret", "bittern", "crane", "limpkin", "European_gallinule",
            "American_coot", "bustard", "ruddy_turnstone", "red-backed_sandpiper",
            "redshank", "dowitcher", "oystercatcher", "pelican", "king_penguin"
        ]
        
        # Completar hasta 1000 clases con nombres genéricos si es necesario
        while len(self.class_names) < 1000:
            self.class_names.append(f"class_{len(self.class_names)}")
        
        logger.info(f"Cargadas {len(self.class_names)} clases de ImageNet")
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocesar imagen para el modelo DenseNet121"""
        try:
            logger.info(f"Preprocesando imagen: {image.size}, modo: {image.mode}")
            
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info("Imagen convertida a RGB")
            
            # Redimensionar a 224x224 (tamaño esperado por DenseNet)
            image = image.resize((224, 224), Image.LANCZOS)
            logger.info("Imagen redimensionada a 224x224")
            
            # Convertir a array numpy con dtype float32 (importante!)
            img_array = np.array(image, dtype=np.float32)
            logger.info(f"Array creado: shape={img_array.shape}, dtype={img_array.dtype}")
            
            # Normalizar píxeles de [0, 255] a [0, 1]
            img_array = img_array / 255.0
            logger.info(f"Píxeles normalizados: min={img_array.min():.3f}, max={img_array.max():.3f}")
            
            # Aplicar normalización ImageNet estándar para DenseNet
            # Mean y std son los valores estándar de ImageNet
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            # Normalizar cada canal
            img_array = (img_array - mean) / std
            logger.info(f"Normalización ImageNet aplicada: min={img_array.min():.3f}, max={img_array.max():.3f}")
            
            # Cambiar de HWC (Height, Width, Channels) a CHW (Channels, Height, Width)
            img_array = img_array.transpose(2, 0, 1)
            logger.info(f"Transpuesta aplicada: shape={img_array.shape}")
            
            # Agregar dimensión de batch [1, 3, 224, 224]
            img_array = np.expand_dims(img_array, axis=0)
            logger.info(f"Dimensión de batch agregada: shape={img_array.shape}")
            
            # Verificar que el dtype sea float32
            if img_array.dtype != np.float32:
                img_array = img_array.astype(np.float32)
                logger.info(f"Dtype convertido a: {img_array.dtype}")
            
            logger.info(f"Preprocesamiento completado: shape={img_array.shape}, dtype={img_array.dtype}")
            return img_array
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}", exc_info=True)
            raise
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Hacer predicción en una imagen"""
        start_time = time.time()
        
        try:
            logger.info(f"Iniciando predicción para imagen de tamaño: {image.size}")
            
            # Preprocesar imagen
            logger.info("Preprocesando imagen...")
            input_data = self._preprocess_image(image)
            logger.info(f"Datos de entrada shape: {input_data.shape}, dtype: {input_data.dtype}")
            
            # Hacer inferencia
            logger.info("Ejecutando inferencia...")
            outputs = self.session.run(
                [self.output_name], 
                {self.input_name: input_data}
            )
            logger.info(f"Inferencia completada. Output shape: {outputs[0].shape}")
            
            # Procesar salida
            predictions = outputs[0][0]  # Remover dimensión de batch
            logger.info(f"Predictions shape después de remover batch: {predictions.shape}")
            
            # Verificar que tenemos el número correcto de clases
            if len(predictions) != len(self.class_names):
                logger.warning(f"Mismatch: predictions={len(predictions)}, class_names={len(self.class_names)}")
            
            # Aplicar softmax para obtener probabilidades
            logger.info("Aplicando softmax...")
            exp_predictions = np.exp(predictions - np.max(predictions))
            probabilities = exp_predictions / np.sum(exp_predictions)
            
            # Obtener top 5 predicciones
            logger.info("Obteniendo top 5 predicciones...")
            top_indices = np.argsort(probabilities)[-5:][::-1]
            
            top_classes = [self.class_names[i] for i in top_indices]
            top_scores = [float(probabilities[i]) for i in top_indices]
            
            processing_time = time.time() - start_time
            logger.info(f"Predicción completada en {processing_time:.3f}s")
            
            return {
                "predictions": top_classes,
                "confidence_scores": top_scores,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}", exc_info=True)
            raise
    
    def is_healthy(self) -> bool:
        """Verificar si el modelo está funcionando"""
        try:
            logger.info("Iniciando health check...")
            
            # Verificar que la sesión esté cargada
            if self.session is None:
                logger.error("Sesión ONNX no inicializada")
                return False
            
            # Verificar que tenemos los nombres de entrada y salida
            if not self.input_name or not self.output_name:
                logger.error(f"Nombres de entrada/salida no válidos: input={self.input_name}, output={self.output_name}")
                return False
            
            # Crear imagen de prueba simple
            test_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            logger.info("Imagen de prueba creada para health check")
            
            # Health check simplificado - solo verificar que el modelo puede hacer inferencia
            try:
                # Preprocesar imagen
                input_data = self._preprocess_image(test_image)
                logger.info(f"Preprocesamiento exitoso en health check: {input_data.shape}")
                
                # Hacer inferencia básica
                outputs = self.session.run(
                    [self.output_name], 
                    {self.input_name: input_data}
                )
                logger.info(f"Inferencia exitosa en health check: {outputs[0].shape}")
                
                # Verificaciones básicas del output
                if outputs and len(outputs) > 0 and outputs[0] is not None:
                    output_shape = outputs[0].shape
                    if len(output_shape) >= 2 and output_shape[1] > 0:  # [batch_size, num_classes]
                        logger.info("Health check: inferencia completada exitosamente")
                        return True
                    else:
                        logger.error(f"Shape de salida inválida: {output_shape}")
                        return False
                else:
                    logger.error("Output de inferencia es None o vacío")
                    return False
                    
            except Exception as pred_error:
                logger.error(f"Error en inferencia de health check: {str(pred_error)}", exc_info=True)
                return False
            
        except Exception as e:
            logger.error(f"Health check falló con excepción: {str(e)}", exc_info=True)
            return False
    
    def simple_health_check(self) -> bool:
        """Health check más simple - solo verifica que el modelo esté cargado"""
        try:
            return (
                self.session is not None and
                self.input_name is not None and
                self.output_name is not None and
                self.input_shape is not None
            )
        except Exception as e:
            logger.error(f"Simple health check falló: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        return {
            "model_name": "DenseNet121",
            "model_path": self.model_path,
            "input_shape": self.input_shape,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "num_classes": len(self.class_names) if self.class_names else 0,
            "providers": self.session.get_providers() if self.session else []
        }