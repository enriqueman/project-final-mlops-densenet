# Imagen base mínima
FROM scratch

# Copiar solo el modelo
COPY densenet121_Opset17.onnx /model/densenet121_Opset17.onnx

# Metadata
LABEL model.name="densenet121"
LABEL model.version="1.0"
LABEL model.format="onnx" 