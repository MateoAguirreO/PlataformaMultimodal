FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py xai.py reflexion.py .
# Fallback horneado en la imagen; en docker-compose se monta como volumen
# de solo lectura para poder reemplazar los modelos sin rebuild.
COPY modelos ./modelos

ENV MODELOS_DIR=/app/modelos
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
