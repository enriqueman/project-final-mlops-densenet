# Usar la imagen base oficial de Lambda para Python
FROM public.ecr.aws/lambda/python:3.9

# Instalar dependencias del sistema necesarias
RUN yum update -y && yum install -y \
    gcc \
    g++ \
    && yum clean all

# Copiar requirements y instalar dependencias Python
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Instalar dependencias en orden específico
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir python-multipart==0.0.6 && \
    pip install --no-cache-dir -r requirements.txt

# Crear directorio para el modelo
RUN mkdir -p ${LAMBDA_TASK_ROOT}/model

# Copiar el código de la aplicación
COPY app.py ${LAMBDA_TASK_ROOT}

# IMPORTANTE: Copiar el modelo ONNX desde el directorio local
# Este archivo debe existir en el directorio de build
COPY densenet121_Opset17.onnx ${LAMBDA_TASK_ROOT}/model/densenet121_Opset17.onnx

# Verificar que el modelo se copió correctamente
RUN ls -la ${LAMBDA_TASK_ROOT}/model/ && \
    echo "Modelo copiado correctamente" && \
    file ${LAMBDA_TASK_ROOT}/model/densenet121_Opset17.onnx

# Verificar la instalación de python-multipart
RUN python -c "import multipart; print('python-multipart installed successfully')"

# Comando para ejecutar la función Lambda
CMD ["app.lambda_handler"]