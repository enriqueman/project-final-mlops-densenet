# Usar la imagen base oficial de Lambda para Python
FROM public.ecr.aws/lambda/python:3.9

# Crear directorio para el modelo
RUN mkdir -p /opt/ml/model

# Copiar el modelo
COPY densenet121_Opset17.onnx /opt/ml/model/densenet121_Opset17.onnx

# Copiar stats si existe
COPY turnkey_stats.yaml /opt/ml/model/turnkey_stats.yaml

# Metadata
LABEL model.name="densenet121"
LABEL model.version="1.0"
LABEL model.format="onnx"

# Comando por defecto (requerido por Lambda)
CMD ["python3"] 