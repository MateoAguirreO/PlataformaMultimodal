"""Backend: evalua una muestra nueva de un participante contra los modelos congelados de
riesgo de ansiedad y depresion (un modelo independiente por eje).

Consume artefactos EXPORTADOS por Psiquiatria/Multimodal_XAI/exportar_plataforma.py: por eje,
la combinacion de ramas con mejor AUC en el analisis por muestra (16-fold x 5, semilla 2026),
reentrenada con todos los participantes. Este repo no entrena nada. Para actualizarlos: correr
ese script y copiar su carpeta modelos_plataforma/ sobre ./modelos/ (o montarla como volumen,
ver docker-compose.yml).

Modelos exportados el 2026-09-30 (AUC de la tabla / con seleccion anidada, en <eje>_meta.json):
  depresion: rostro (AU) + voz (wav2vec2, micro-ventanas)   0.701 / 0.558
  ansiedad:  rostro (AU) + voz (eGeMAPS)                    0.682 / 0.579
Los de la tabla son optimistas (mejor de 127 combinaciones); los de seleccion anidada son la
estimacion honesta. Fusion: soft vote de las ramas disponibles (ver ramas.py).

Acceso: todos los endpoints de evaluacion exigen sesion iniciada con rol admin o clinico
(ver seguridad.py y rutas_auth.py); /health queda publico para el chequeo de Docker.

Uso local (sin Docker):
  pip install -r requirements.txt
  uvicorn app:app --host 127.0.0.1 --port 8000
(solo en localhost: el acceso de los usuarios es por el frontend/BFF, que agrega la IP
real del cliente para la auditoria y el limite de intentos de login)
"""
import concurrent.futures
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

import db
import extraction
import ramas
import reflexion as reflexion_mod
import report_pdf
import seguridad
import voz_embedding
import xai
from rutas_auth import router_admin, router_auth

DIR_MODELOS = Path(os.environ.get("MODELOS_DIR", Path(__file__).resolve().parent / "modelos"))

app = FastAPI(title="Riesgo ansiedad/depresion — backend de evaluacion")
app.include_router(router_auth)
app.include_router(router_admin)
_MODELOS: dict = {}


def _evaluar(accion):
    """Dependencia de las rutas de evaluacion: sesion + rol (admin/clinico) + auditoria."""
    return [Depends(seguridad.acceso_evaluacion(accion))]


def _cargar_modelos():
    _MODELOS.update(ramas.cargar(DIR_MODELOS))


@app.on_event("startup")
def startup():
    db.configurar()
    with db.SessionLocal() as dbs:
        seguridad.crear_admin_inicial(dbs)
    _cargar_modelos()
    if not _MODELOS:
        raise RuntimeError(
            f"no hay modelos en {DIR_MODELOS} -- copia ahi el contenido de "
            f"Multimodal_XAI/modelos_plataforma/ del repo Psiquiatria (o corre "
            f"exportar_plataforma.py alla primero)")
    for dx, m in _MODELOS.items():
        print(f"[backend] {dx}: {'+'.join(m['meta']['combinacion'])}  (modelos desde {DIR_MODELOS})")
    p = ramas.extraccion_voz(_MODELOS)
    if p and os.environ.get("PRECARGAR_VOZ", "true").lower() == "true":
        voz_embedding.precargar(p["modelo"])   # en segundo plano: ~1.2 GB la primera vez


class Muestra(BaseModel):
    audio_segments: list[dict[str, float]] = Field(
        ..., description="1 dict por segmento de audio, features eGeMAPS (nombre->valor)")
    video_features: dict[str, float] = Field(
        ..., description="1 dict con las features de rostro del participante (nombre->valor)")
    voz_embedding: list[float] | None = Field(
        None, description="embedding de voz del participante (POST /voz/embedding con su WAV); "
                          "sin el, la rama de voz por embedding queda fuera de la fusion")


def _prediccion(dx, muestra):
    try:
        return ramas.evaluar(_MODELOS[dx], muestra)[0]
    except ValueError as e:
        raise HTTPException(422, f"[{dx}] {e}") from e


def _explicacion(dx, muestra):
    try:
        return xai.explicar(dx, _MODELOS, muestra)
    except ValueError as e:
        raise HTTPException(422, f"[{dx}] {e}") from e


@app.post("/predict", dependencies=_evaluar("predict"))
def predict(muestra: Muestra):
    if not muestra.audio_segments:
        raise HTTPException(422, "audio_segments no puede venir vacio")
    return {dx: _prediccion(dx, muestra) for dx in _MODELOS}


@app.post("/voz/embedding", dependencies=_evaluar("voz_embedding"))
def voz_embedding_wav(audio: UploadFile = File(..., description="WAV con solo la voz del participante, sin ruido")):
    """Embedding de voz (rama voz_microventanas) a partir del audio del participante. El front
    lo pide una vez y lo manda en voz_embedding de /predict, /explain y /report."""
    p = ramas.extraccion_voz(_MODELOS)
    if p is None:
        raise HTTPException(404, "ningun modelo cargado usa embedding de voz")
    with tempfile.TemporaryDirectory() as tmp:
        ruta = _ruta_subida(Path(tmp), audio, "audio", ".wav")
        with open(ruta, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        try:
            return voz_embedding.calcular(ruta, p)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e


def _verificar_o_none(payload):
    """Corre el loop jr/senior; si Gemini falla por CUALQUIER motivo (cuota
    agotada, red, modelo no disponible) degrada con gracia en vez de tumbar
    el endpoint -- la prediccion + SHAP siguen siendo utiles sin esto."""
    if not reflexion_mod.reflexion_disponible():
        return None, "verificacion jr/senior no disponible: falta GEMINI_API_KEY en el backend"
    try:
        return reflexion_mod.verificar(payload), None
    except Exception as e:  # noqa: BLE001 -- cualquier falla de Gemini es no-fatal aqui
        return None, f"verificacion jr/senior fallo ({type(e).__name__}): {e}"


STAGGER_SECONDS = 4  # separacion entre el arranque de cada eje en /report -- se
                      # corren en paralelo pero sin este delay las 2 primeras
                      # llamadas ("jr" de cada eje) le pegan a Gemini en el mismo
                      # instante y pueden chocar contra el limite de requests/minuto
                      # (distinto del limite diario, que el fallback de reflexion.py
                      # ya cubre).


def _verificar_con_stagger(payload, delay):
    if delay:
        time.sleep(delay)
    return _verificar_o_none(payload)


@app.post("/explain", dependencies=_evaluar("explain"))
def explain(muestra: Muestra, eje: str):
    """SHAP local de la muestra + (si hay GEMINI_API_KEY) el proceso completo
    de verificacion jr/senior -- expuesto para que el front detalle el
    proceso interno de la IA (plataforma academica, no comercial)."""
    if eje not in _MODELOS:
        raise HTTPException(404, f"eje '{eje}' no disponible (cargados: {list(_MODELOS.keys())})")
    if not muestra.audio_segments:
        raise HTTPException(422, "audio_segments no puede venir vacio")

    payload = _explicacion(eje, muestra)
    resultado, nota = _verificar_o_none(payload)
    return {"shap": payload, "reflexion": resultado, "reflexion_nota": nota}


@app.post("/report", dependencies=_evaluar("report"))
def report(muestra: Muestra):
    """Informe en PDF: prediccion + SHAP + verificacion jr/senior, ambos ejes.
    Recalcula todo (no cachea /predict ni /explain).

    SHAP es rapido (<0.1s/eje) pero la verificacion jr/senior son varias
    llamadas secuenciales a Gemini por eje (~1-4 min/eje medido en la
    practica) -- como ansiedad y depresion son independientes entre si, se
    corren en paralelo (I/O-bound, threads alcanzan) en vez de una detras de
    la otra, con un pequeno stagger entre el arranque de cada una (ver
    STAGGER_SECONDS) para no chocar contra el limite de requests/minuto de
    Gemini con las 2 primeras llamadas al mismo tiempo."""
    if not muestra.audio_segments:
        raise HTTPException(422, "audio_segments no puede venir vacio")
    payloads = {dx: _explicacion(dx, muestra) for dx in _MODELOS}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        futuros = {dx: pool.submit(_verificar_con_stagger, payload, i * STAGGER_SECONDS)
                   for i, (dx, payload) in enumerate(payloads.items())}
        resultados = {dx: fut.result() for dx, fut in futuros.items()}
    secciones = {dx: {"shap": payloads[dx], "reflexion": resultados[dx][0], "reflexion_nota": resultados[dx][1]}
                 for dx in _MODELOS}
    pdf_bytes = report_pdf.generar(secciones)
    return Response(content=pdf_bytes, media_type="application/pdf",
                     headers={"Content-Disposition": 'attachment; filename="informe_riesgo.pdf"'})


def _ruta_subida(workdir: Path, archivo: UploadFile, base: str, ext_defecto: str) -> Path:
    """Nombre fijo + solo la extension del archivo subido: usar el nombre que manda el
    cliente permitiria escribir fuera de workdir (p. ej. "../../app/app.py" o una ruta
    absoluta, que Path / descarta el prefijo)."""
    ext = Path(archivo.filename or "").suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,5}", ext):
        ext = ext_defecto
    return workdir / f"{base}{ext}"


def _muestra_desde_media(audio: UploadFile, video: UploadFile, workdir: Path) -> Muestra:
    """mp4+wav crudos -> Muestra (audio_segments + video_features + voz_embedding), corriendo
    Demucs+eGeMAPS, frames+py-feat y wav2vec2 (modo DEMO, ~4 min en GPU -- ver
    extraction.py para las notas de fidelidad vs. entrenamiento). El wav debe ser el
    recortado (solo la voz del participante): las ramas de voz se entrenaron asi."""
    audio_path = _ruta_subida(workdir, audio, "audio", ".wav")
    video_path = _ruta_subida(workdir, video, "video", ".mp4")
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    denoised = extraction.denoise_audio(audio_path, workdir)
    audio_segments = extraction.audio_to_segments(denoised)
    video_features = extraction.video_to_features(video_path, workdir)
    return Muestra(audio_segments=audio_segments, video_features=video_features,
                   voz_embedding=_embedding_o_none(denoised))


def _embedding_o_none(ruta_audio):
    """Embedding de voz del audio ya sin ruido, si algun eje lo usa. Si el modelo de voz no
    esta disponible, esa rama queda fuera de la fusion (la respuesta lo dice) en vez de fallar."""
    p = ramas.extraccion_voz(_MODELOS)
    if p is None:
        return None
    try:
        return voz_embedding.calcular(ruta_audio, p)["embedding"]
    except RuntimeError as e:
        print(f"[backend] embedding de voz omitido: {e}", flush=True)
        return None


@app.post("/predict_raw", dependencies=_evaluar("predict_raw"))
def predict_raw(audio: UploadFile = File(..., description="wav recortado: solo la voz del participante"),
                 video: UploadFile = File(..., description="mp4 de la entrevista")):
    """Como /predict, pero a partir del mp4+wav crudos (modo demo: Demucs+eGeMAPS
    para audio, 1fps+py-feat para rostro, ambos en GPU -- ~4 min end-to-end)."""
    with tempfile.TemporaryDirectory() as tmp:
        muestra = _muestra_desde_media(audio, video, Path(tmp))
        return predict(muestra)


@app.post("/explain_raw", dependencies=_evaluar("explain_raw"))
def explain_raw(eje: str,
                 audio: UploadFile = File(...), video: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        muestra = _muestra_desde_media(audio, video, Path(tmp))
        return explain(muestra, eje)


@app.post("/report_raw", dependencies=_evaluar("report_raw"))
def report_raw(audio: UploadFile = File(...), video: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        muestra = _muestra_desde_media(audio, video, Path(tmp))
        return report(muestra)


@app.get("/health")
def health():
    return {"status": "ok", "ejes_cargados": list(_MODELOS.keys()),
            "modelos": {dx: "+".join(m["meta"]["combinacion"]) for dx, m in _MODELOS.items()},
            "voz_embedding": voz_embedding.estado() if ramas.extraccion_voz(_MODELOS) else "no se usa",
            "reflexion_disponible": reflexion_mod.reflexion_disponible()}


@app.get("/schema", dependencies=[Depends(seguridad.puede_evaluar)])
def schema():
    """Entradas que espera cada eje -- para armar el CSV/JSON de entrada."""
    return {dx: {"combinacion": m["meta"]["combinacion"],
                 "entradas": {r["entrada"]: r.get("features") or
                              {"dim": r["dim"], "como": "POST /voz/embedding con el WAV del participante"}
                              for r in m["meta"]["ramas"].values()}}
            for dx, m in _MODELOS.items()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
