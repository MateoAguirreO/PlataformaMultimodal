"""Embedding de voz de la rama `voz_microventanas` (modelo de depresión).

Reproduce la extracción con la que se entrenó (réplica determinista "R" de
Multimodal/experimento_embeddings/replicar_emb_psiquiatria.py):
  1. audio a 16 kHz mono, normalizado por pico;
  2. ventanas de 5 s con salto de 2.5 s (mínimo 1 s; las cortas se rellenan con ceros);
  3. de cada ventana entran solo los primeros 3000 samples (0.19 s: el recorte
     max_length=3000 del código original de Psiquiatria);
  4. última capa de wav2vec2-large-robust, media de sus frames, y media sobre ventanas.
Los parámetros vienen del meta del modelo (`extraccion`), que escribe el exportador.

El audio debe ser SOLO la voz del participante (recortado) y sin ruido, como el de
entrenamiento: sobre el audio crudo de la entrevista esta representación no funciona.

El modelo (~1.2 GB) se descarga de Hugging Face la primera vez y queda en caché. Se carga en
un hilo al arrancar el backend (precargar) para que la primera evaluación no espere.
"""
import threading

import numpy as np

_lock = threading.Lock()
_estado = {"id": None, "modelo": None, "extractor": None, "dispositivo": None, "error": None,
           "cargando": False}
LOTE = 512  # ventanas por pasada: una sola para audios de hasta ~21 min, como en el entrenamiento
            # (partir el lote cambia el redondeo en fp16 y el ExtraTrees es sensible a eso)


def _cargar(model_id: str):
    with _lock:
        if _estado["modelo"] is not None and _estado["id"] == model_id:
            return
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()   # el checkpoint trae cabezas de preentrenamiento que no se usan
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        modelo = Wav2Vec2Model.from_pretrained(model_id).eval().to(dispositivo)
        if dispositivo == "cuda":
            modelo = modelo.half()       # como en el entrenamiento (fp16 en GPU)
        _estado.update(id=model_id, modelo=modelo, dispositivo=dispositivo, error=None,
                       extractor=Wav2Vec2FeatureExtractor.from_pretrained(model_id))


def precargar(model_id: str) -> None:
    """Carga el modelo en segundo plano; un fallo (p. ej. sin internet la primera vez) queda
    en estado() y se reintenta en la siguiente evaluación."""
    def tarea():
        _estado["cargando"] = True
        try:
            _cargar(model_id)
            print(f"[voz_embedding] {model_id} listo en {_estado['dispositivo']}", flush=True)
        except Exception as e:  # noqa: BLE001 -- el backend sigue; esa rama queda no disponible
            _estado["error"] = f"{type(e).__name__}: {e}"
            print(f"[voz_embedding] no se pudo cargar {model_id}: {_estado['error']}", flush=True)
        finally:
            _estado["cargando"] = False
    threading.Thread(target=tarea, daemon=True).start()


def estado() -> str:
    if _estado["modelo"] is not None:
        return f"listo ({_estado['dispositivo']})"
    if _estado["cargando"]:
        return "cargando"
    return f"error: {_estado['error']}" if _estado["error"] else "sin cargar"


def ventanas(y: np.ndarray, ventana: int, salto: int, minimo: int) -> list:
    segs, s = [], 0
    while s < len(y):
        c = y[s:s + ventana]
        if len(c) >= minimo:
            segs.append(np.pad(c, (0, ventana - len(c))) if len(c) < ventana else c)
        s += salto
    return segs


def calcular(ruta_audio, p: dict) -> dict:
    """ruta_audio -> {"embedding": lista de p-dim, "n_ventanas", "duracion_s"}.
    `p` es el dict `extraccion` del meta del modelo.
    ValueError: el audio no se puede usar. RuntimeError: el modelo de voz no está disponible."""
    import librosa
    import torch

    try:
        y, _ = librosa.load(str(ruta_audio), sr=p["sr"], mono=True)
    except Exception as e:  # noqa: BLE001 -- formatos corruptos o no soportados
        raise ValueError(f"no se pudo leer el audio ({type(e).__name__}); sube un WAV") from e
    if len(y) < p["minimo"]:
        raise ValueError(f"el audio dura {len(y) / p['sr']:.1f} s; se necesita al menos "
                         f"{p['minimo'] / p['sr']:.0f} s de voz del participante")
    y = y / (np.max(np.abs(y)) + 1e-9)
    segs = ventanas(y, p["ventana"], p["salto"], p["minimo"])
    try:
        _cargar(p["modelo"])
    except Exception as e:  # noqa: BLE001 -- p. ej. sin internet para descargarlo la primera vez
        _estado["error"] = f"{type(e).__name__}: {e}"
        raise RuntimeError(f"el modelo de voz ({p['modelo']}) no está disponible: {_estado['error']}") from e
    fe, modelo, dispositivo = _estado["extractor"], _estado["modelo"], _estado["dispositivo"]
    x = fe(segs, sampling_rate=p["sr"], return_tensors="pt", padding="max_length",
           max_length=p["recorte"], truncation=True).input_values.to(dispositivo)
    if dispositivo == "cuda":
        x = x.half()
    with torch.no_grad():
        E = np.concatenate([modelo(x[i:i + LOTE]).last_hidden_state.float().mean(dim=1).cpu().numpy()
                            for i in range(0, len(x), LOTE)])
    return {"embedding": E.mean(axis=0).astype(float).tolist(), "n_ventanas": len(segs),
            "duracion_s": round(len(y) / p["sr"], 1)}
