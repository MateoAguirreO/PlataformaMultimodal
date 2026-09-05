"""Backend: evalua una muestra nueva (audio segmentado + rostro de un
participante) contra los modelos congelados de riesgo ansiedad/depresion.

Consume artefactos EXPORTADOS del repo de tesis (AnalisisMultimodal /
pipeline_multimodal_xai/train_final.py) -- este repo no entrena nada, solo
sirve modelos ya congelados. Para actualizarlos: correr train_final.py en el
repo de tesis y copiar el contenido de pipeline_multimodal_xai/modelos/ sobre
./modelos/ aqui (o montar esa carpeta como volumen -- ver docker-compose.yml).

Combina voz+rostro con soft_vote (promedio simple) -- arquitectura ganadora
documentada en REPORTE_multimodal.md SS3 del repo de tesis, con los modelos
ACTUALES (0.67-0.68 AUC). Pendiente reemplazar cuando lleguen los modelos de
0.8 AUC.

No hace SHAP ni informe en PDF todavia -- ver README, seccion "pendiente".

Uso local (sin Docker):
  pip install -r requirements.txt
  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

import extraction
import reflexion as reflexion_mod
import report_pdf
import xai

DIR_MODELOS = Path(os.environ.get("MODELOS_DIR", Path(__file__).resolve().parent / "modelos"))
EJES = ("ansiedad", "depresion")

app = FastAPI(title="Riesgo ansiedad/depresion — backend de evaluacion")
_MODELOS: dict = {}


def _agg(valores, modo):
    if modo == "mean":
        return float(np.mean(valores))
    if modo == "median":
        return float(np.median(valores))
    if modo.startswith("p"):
        return float(np.percentile(valores, float(modo[1:])))
    raise ValueError(modo)


def _cargar_modelos():
    for dx in EJES:
        meta_path = DIR_MODELOS / f"{dx}_meta.json"
        if not meta_path.exists():
            continue
        _MODELOS[dx] = {
            "audio": joblib.load(DIR_MODELOS / f"{dx}_audio.joblib"),
            "video": joblib.load(DIR_MODELOS / f"{dx}_video.joblib"),
            "audio_bg": joblib.load(DIR_MODELOS / f"{dx}_audio_bg.joblib"),
            "video_bg": joblib.load(DIR_MODELOS / f"{dx}_video_bg.joblib"),
            "meta": json.loads(meta_path.read_text(encoding="utf-8")),
        }


@app.on_event("startup")
def startup():
    _cargar_modelos()
    if not _MODELOS:
        raise RuntimeError(
            f"no hay modelos en {DIR_MODELOS} -- copia ahi el contenido de "
            f"pipeline_multimodal_xai/modelos/ del repo de tesis (o corre "
            f"train_final.py alla primero)")
    print(f"[backend] ejes cargados: {list(_MODELOS.keys())}  (modelos desde {DIR_MODELOS})")


class Muestra(BaseModel):
    audio_segments: list[dict[str, float]] = Field(
        ..., description="1 dict por segmento de audio, features eGeMAPS (nombre->valor)")
    video_features: dict[str, float] = Field(
        ..., description="1 dict con las features de rostro del participante (nombre->valor)")


def _faltantes(requeridas, disponibles):
    return [f for f in requeridas if f not in disponibles]


def _score_audio(dx, segments):
    meta = _MODELOS[dx]["meta"]
    feats = meta["audio_features"]
    for i, s in enumerate(segments):
        falt = _faltantes(feats, s)
        if falt:
            raise HTTPException(422, f"[{dx}] segmento {i}: faltan features de audio "
                                      f"{falt[:5]}{'...' if len(falt) > 5 else ''}")
    X = np.array([[s[f] for f in feats] for s in segments])
    p = _MODELOS[dx]["audio"].predict_proba(X)[:, 1]
    return _agg(p, meta.get("audio_agg", "mean"))


def _score_video(dx, video_features):
    meta = _MODELOS[dx]["meta"]
    feats = meta["video_features"]
    falt = _faltantes(feats, video_features)
    if falt:
        raise HTTPException(422, f"[{dx}] faltan features de rostro "
                                  f"{falt[:5]}{'...' if len(falt) > 5 else ''}")
    X = np.array([[video_features[f] for f in feats]])
    return float(_MODELOS[dx]["video"].predict_proba(X)[0, 1])


@app.post("/predict")
def predict(muestra: Muestra):
    if not muestra.audio_segments:
        raise HTTPException(422, "audio_segments no puede venir vacio")
    out = {}
    for dx in _MODELOS:
        s_audio = _score_audio(dx, muestra.audio_segments)
        s_video = _score_video(dx, muestra.video_features)
        p = (s_audio + s_video) / 2.0  # soft_vote: ganador actual en los dos ejes
        meta = _MODELOS[dx]["meta"]
        out[dx] = {
            "p_riesgo": round(p, 4),
            "clase": "riesgo" if p >= 0.5 else "sin riesgo",
            "score_audio": round(s_audio, 4),
            "score_video": round(s_video, 4),
            "n_segmentos_audio": len(muestra.audio_segments),
            "modelo": {"audio": meta["audio_config"], "video": meta["video_config"],
                       "fusion": "soft_vote", "auc_validado": meta["auc_esperado"]},
        }
    return out


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


@app.post("/explain")
def explain(muestra: Muestra, eje: str):
    """SHAP local de la muestra + (si hay GEMINI_API_KEY) el proceso completo
    de verificacion jr/senior -- expuesto para que el front detalle el
    proceso interno de la IA (plataforma academica, no comercial)."""
    if eje not in _MODELOS:
        raise HTTPException(404, f"eje '{eje}' no disponible (cargados: {list(_MODELOS.keys())})")
    if not muestra.audio_segments:
        raise HTTPException(422, "audio_segments no puede venir vacio")

    payload = xai.explicar(eje, _MODELOS, muestra.audio_segments, muestra.video_features)
    resultado, nota = _verificar_o_none(payload)
    return {"shap": payload, "reflexion": resultado, "reflexion_nota": nota}


@app.post("/report")
def report(muestra: Muestra):
    """Informe en PDF: prediccion + SHAP + verificacion jr/senior, ambos ejes.
    Recalcula todo (no cachea /predict ni /explain) -- tarda lo mismo que
    llamar /explain dos veces (una por eje)."""
    if not muestra.audio_segments:
        raise HTTPException(422, "audio_segments no puede venir vacio")
    secciones = {}
    for dx in _MODELOS:
        payload = xai.explicar(dx, _MODELOS, muestra.audio_segments, muestra.video_features)
        resultado, nota = _verificar_o_none(payload)
        secciones[dx] = {"shap": payload, "reflexion": resultado, "reflexion_nota": nota}
    pdf_bytes = report_pdf.generar(secciones)
    return Response(content=pdf_bytes, media_type="application/pdf",
                     headers={"Content-Disposition": 'attachment; filename="informe_riesgo.pdf"'})


def _muestra_desde_media(audio: UploadFile, video: UploadFile, workdir: Path) -> Muestra:
    """mp4+wav crudos -> Muestra (audio_segments + video_features), corriendo
    Demucs+eGeMAPS y frames+py-feat (modo DEMO, ~4 min en GPU -- ver
    extraction.py para las notas de fidelidad vs. entrenamiento)."""
    audio_path = workdir / (audio.filename or "audio.wav")
    video_path = workdir / (video.filename or "video.mp4")
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    denoised = extraction.denoise_audio(audio_path, workdir)
    audio_segments = extraction.audio_to_segments(denoised)
    video_features = extraction.video_to_features(video_path, workdir)
    return Muestra(audio_segments=audio_segments, video_features=video_features)


@app.post("/predict_raw")
def predict_raw(audio: UploadFile = File(..., description="wav de la entrevista"),
                 video: UploadFile = File(..., description="mp4 de la entrevista")):
    """Como /predict, pero a partir del mp4+wav crudos (modo demo: Demucs+eGeMAPS
    para audio, 1fps+py-feat para rostro, ambos en GPU -- ~4 min end-to-end)."""
    with tempfile.TemporaryDirectory() as tmp:
        muestra = _muestra_desde_media(audio, video, Path(tmp))
        return predict(muestra)


@app.post("/explain_raw")
def explain_raw(eje: str,
                 audio: UploadFile = File(...), video: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        muestra = _muestra_desde_media(audio, video, Path(tmp))
        return explain(muestra, eje)


@app.post("/report_raw")
def report_raw(audio: UploadFile = File(...), video: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        muestra = _muestra_desde_media(audio, video, Path(tmp))
        return report(muestra)


@app.get("/health")
def health():
    return {"status": "ok", "ejes_cargados": list(_MODELOS.keys()),
            "reflexion_disponible": reflexion_mod.reflexion_disponible()}


@app.get("/schema")
def schema():
    """Nombres de features que espera cada eje -- para armar el CSV/JSON de entrada."""
    return {dx: {"audio_features": m["meta"]["audio_features"],
                 "video_features": m["meta"]["video_features"]}
            for dx, m in _MODELOS.items()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
