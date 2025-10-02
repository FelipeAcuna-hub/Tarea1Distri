# Usamos la imagen Alpine para evitar problemas de red/proxy
FROM python:3.11-alpine

WORKDIR /app

# Instalar dependencias del sistema (para asyncpg/psycopg)
# Alpine usa 'apk add'
RUN apk update && \
    apk add --no-cache gcc musl-dev postgresql-dev

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- COPIA DE ARCHIVOS ESPECÍFICOS ---

# Crear la carpeta /data explícitamente y copiar el CSV
RUN mkdir -p /data 
COPY train.csv /data/train.csv 

# Copia el script Python de carga de datos
COPY data_loader.py .

CMD ["python", "data_loader.py"]
