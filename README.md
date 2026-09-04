# PlataformaMultimodal

Microservicio para evaluar una muestra nueva contra los modelos de riesgo de
ansiedad/depresión (fusión voz+rostro, `soft_vote`). Proyecto **separado** del
repo de tesis ([AnalisisMultimodal](https://github.com/MateoAguirreO/AnalisisMultimodal))
a propósito: este solo *sirve* modelos ya entrenados, no hace investigación
(sin datos crudos, sin notebooks de experimentos, sin literatura).

- **Backend** (`backend/`): Python + FastAPI. Carga los `.joblib` congelados
  y responde `POST /predict`.
- **Frontend** (`frontend/`): Node.js + Express. Sirve la interfaz estática y
  hace de proxy `/api/*` → backend (mismo origen para el navegador, sin CORS).
- Cada uno en su propio contenedor; `docker-compose.yml` los orquesta.

## Con los modelos actuales (0.67-0.68 AUC)

`backend/modelos/` viene con una **copia congelada** de los modelos actuales
del repo de tesis (`pipeline_multimodal_xai/train_final.py`). **Pendiente:**
reemplazarlos cuando lleguen los 3 modelos de 0.8 AUC — ver la nota en
`backend/app.py`.

### Actualizar los modelos

```bash
# en el repo de tesis
python pipeline_multimodal_xai/train_final.py

# copiar el export aca
cp <repo-tesis>/pipeline_multimodal_xai/modelos/* backend/modelos/
```

Con Docker Compose los modelos se montan como volumen (`backend/modelos` →
`/app/modelos`), así que un restart del contenedor basta — no hace falta
rebuild.

## Correr con Docker (recomendado — así se despliega)

```bash
docker compose up --build
# abrir http://localhost:3000
```

Solo el frontend expone puerto al host (3000); el backend vive solo en la red
interna de Docker.

## Correr en local sin Docker (dev)

```bash
# terminal 1 — backend
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000

# terminal 2 — frontend
cd frontend
npm install
BACKEND_URL=http://localhost:8000 npm start
# abrir http://localhost:3000
```

## API del backend

- `GET /health` → `{"status": "ok", "ejes_cargados": [...]}`
- `GET /schema` → nombres exactos de features que espera cada eje.
- `POST /predict` — body:
  ```json
  {
    "audio_segments": [{"F0semitoneFrom27.5Hz_sma3nz_amean": 1.2, "...": "..."}],
    "video_features": {"AU01_mean": 0.3, "anger_mean": 0.02, "...": "..."}
  }
  ```
  → `{"ansiedad": {...}, "depresion": {...}}` con `p_riesgo`, `clase`,
  `score_audio`, `score_video`, `modelo`.

Desde el frontend, las mismas rutas van bajo `/api/...` (el proxy de Express
reescribe `/api/predict` → `backend:8000/predict`).

## Pendiente

1. SHAP local sobre la muestra nueva evaluada (hoy solo predicción).
2. Enchufar el loop de reflexión jr/senior (`xai_reflexion.py` en el repo de
   tesis) sobre esa explicación.
3. Informe en PDF.
4. Reemplazar los modelos actuales por los de 0.8 AUC cuando lleguen.
