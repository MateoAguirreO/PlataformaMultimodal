# PlataformaMultimodal

## Requisitos

- Docker (recomendado), o Python 3.11 + Node 20 para correr sin Docker.
- Opcional: `GEMINI_API_KEY` para la verificación jr/senior (sin ella, todo lo demás funciona igual).

## Configurar la API key (opcional)

```bash
cp .env.example .env
# editar .env y pegar la key real
```

## Correr con Docker

```bash
docker compose up --build
```

Abrir **http://localhost:3000**

## Correr en local sin Docker

```bash
# terminal 1 — backend
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000

# terminal 2 — frontend
cd frontend
npm install
BACKEND_URL=http://localhost:8000 npm start
```

Abrir **http://localhost:3000**

## Probar rápido

Sube los CSV de `ejemplos/` (`positivo_audio.csv`+`positivo_video.csv`, o
`negativo_audio.csv`+`negativo_video.csv`) en la interfaz y dale a Evaluar.
