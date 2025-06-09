# Usar la imagen base oficial de Lambda para Python
FROM public.ecr.aws/lambda/python:3.9

# Instalar dependencias del sistema si es necesario
RUN yum update -y && yum install -y \
    gcc \
    g++ \
    && yum clean all

# Copiar requirements y instalar dependencias Python
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Crear directorio para modelos temporales
RUN mkdir -p /tmp

# Copiar el código de la aplicación
COPY app.py ${LAMBDA_TASK_ROOT}

# Comando para ejecutar la función Lambda
CMD ["app.lambda_handler"]