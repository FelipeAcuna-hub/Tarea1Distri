FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código de la aplicación (app.py)
COPY app.py .

# Exponer el puerto
EXPOSE 8000

# Comando para iniciar la aplicación (Uvicorn/FastAPI)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
