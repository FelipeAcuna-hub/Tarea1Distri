FROM python:3.11-slim
WORKDIR /app
# Instalar aiohttp para peticiones HTTP y asyncpg para la DB
RUN pip install aiohttp asyncpg
COPY traffic_gen.py .
CMD ["python", "traffic_gen.py"]
