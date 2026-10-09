FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py xai.py reflexion.py report_pdf.py extraction.py ramas.py voz_embedding.py ./
COPY db.py seguridad.py rutas_auth.py gestionar_usuarios.py ./
# Fallback horneado en la imagen; en docker-compose se monta como volumen
# de solo lectura para poder reemplazar los modelos sin rebuild.
COPY modelos ./modelos
# Base de usuarios, sesiones y auditoria (SQLite); en docker-compose es un volumen.
RUN mkdir -p /app/datos

ENV MODELOS_DIR=/app/modelos
ENV DATOS_DIR=/app/datos
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
