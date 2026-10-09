"""Modelos por eje: carga, entrada de cada rama, predicción y fusión.

Cada eje tiene un `<eje>_meta.json` y un `<eje>_<rama>.joblib` por rama, generados por
Psiquiatria/Multimodal_XAI/exportar_plataforma.py (la combinación de ramas con mejor AUC en el
análisis por muestra, reentrenada con todos los participantes). Los ejes son independientes:
cada uno tiene su propia combinación.

Entradas de una muestra y rama que alimentan:
  video_features   -> ramas de rostro (au): 66 features del participante
  audio_segments   -> eGeMAPS: media de los segmentos (como en el entrenamiento)
  voz_embedding    -> wav2vec2 micro-ventanas: vector calculado con voz_embedding.py

Fusión: soft vote = promedio de las probabilidades de las ramas DISPONIBLES para la muestra,
igual que en el entrenamiento (allí el participante sin audio quedaba solo con rostro). Si
falta una rama, la respuesta lo dice explícitamente.
"""
import json
from pathlib import Path

import joblib
import numpy as np

MOTIVO_FALTA = {
    "audio_segments": "la muestra no trae segmentos de audio (eGeMAPS)",
    "voz_embedding": "falta el audio del participante (WAV) para calcular el embedding de voz",
    "video_features": "la muestra no trae features de rostro",
}


def cargar(directorio: Path) -> dict:
    modelos = {}
    for ruta_meta in sorted(directorio.glob("*_meta.json")):
        meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
        if "ramas" not in meta:
            raise RuntimeError(f"{ruta_meta.name} tiene el formato anterior (voz+rostro por segmento): "
                               f"exporta los modelos con Multimodal_XAI/exportar_plataforma.py")
        eje = meta["eje"]
        modelos[eje] = {"meta": meta,
                        "ramas": {b: joblib.load(directorio / f"{eje}_{b}.joblib") for b in meta["combinacion"]}}
    return modelos


def extraccion_voz(modelos: dict) -> dict | None:
    """Parámetros de extracción del embedding de voz, si algún eje lo usa."""
    for m in modelos.values():
        for r in m["meta"]["ramas"].values():
            if r["entrada"] == "voz_embedding":
                return r["extraccion"]
    return None


def _faltan(requeridas, disponibles, donde):
    falt = [f for f in requeridas if f not in disponibles]
    if falt:
        raise ValueError(f"{donde}: faltan features {falt[:5]}{'...' if len(falt) > 5 else ''}")


def entrada(rama: dict, muestra) -> np.ndarray | None:
    """Vector de entrada de la rama para esta muestra, o None si la muestra no la trae.
    ValueError si la trae incompleta o mal formada."""
    origen = rama["entrada"]
    if origen == "video_features":
        if not muestra.video_features:
            return None
        _faltan(rama["features"], muestra.video_features, "rostro")
        return np.array([muestra.video_features[f] for f in rama["features"]], dtype=float)
    if origen == "audio_segments":
        if not muestra.audio_segments:
            return None
        for i, s in enumerate(muestra.audio_segments):
            _faltan(rama["features"], s, f"segmento de audio {i}")
        return np.array([[s[f] for f in rama["features"]] for s in muestra.audio_segments], dtype=float).mean(axis=0)
    if origen == "voz_embedding":
        if muestra.voz_embedding is None:
            return None
        x = np.asarray(muestra.voz_embedding, dtype=float)
        if x.shape != (rama["dim"],) or not np.isfinite(x).all():
            raise ValueError(f"voz_embedding debe tener {rama['dim']} valores finitos (tiene {x.size})")
        return x
    raise ValueError(f"entrada desconocida: {origen}")


def probabilidad(bundle: dict, x: np.ndarray) -> float:
    x = np.where(np.isnan(x), bundle["medianas"], x)       # misma imputación que en el entrenamiento
    return float(bundle["pipeline"].predict_proba(x[None, :])[0, 1])


def resumen_modelo(meta: dict) -> dict:
    return {"combinacion": "+".join(meta["combinacion"]), "fusion": meta["fusion"],
            "auc_combinacion_cv": meta["auc"]["combinacion_cv"],
            "auc_seleccion_anidada": meta["auc"]["seleccion_anidada"],
            "n_participantes": meta["n_participantes"], "etiqueta": meta["etiqueta"],
            "umbral": meta.get("umbral"), "advertencias": meta["advertencias"]}


def evaluar(modelo_eje: dict, muestra) -> tuple[dict, dict]:
    """-> (resultado para la API, {rama: (x, prob)} de las ramas disponibles)."""
    meta = modelo_eje["meta"]
    filas, usadas = [], {}
    for b in meta["combinacion"]:
        r = meta["ramas"][b]
        x = entrada(r, muestra)
        fila = {"rama": b, "nombre": r["nombre"], "modalidad": r["modalidad"], "disponible": x is not None,
                "prob": None}
        if x is None:
            fila["motivo"] = MOTIVO_FALTA[r["entrada"]]
        else:
            p = probabilidad(modelo_eje["ramas"][b], x)
            fila["prob"] = round(p, 4)
            usadas[b] = (x, p)
        filas.append(fila)
    if not usadas:
        raise ValueError(f"[{meta['eje']}] la muestra no trae ninguna de las entradas del modelo")
    p = float(np.mean([p for _, p in usadas.values()]))
    clave = "+".join(b for b in meta["combinacion"] if b in usadas)
    rend = meta.get("subconjuntos", {}).get(clave)
    resultado = {"p_riesgo": round(p, 4), "clase": "riesgo" if p >= 0.5 else "sin riesgo",
                 "ramas": filas, "n_segmentos_audio": len(muestra.audio_segments or []),
                 # si falta una rama, la prediccion la hace otro modelo (el subconjunto disponible),
                 # con su propio AUC y su propia sensibilidad/especificidad
                 "parcial": len(usadas) < len(meta["combinacion"]),
                 "rendimiento": {"ramas": clave, **rend} if rend else None,
                 "modelo": resumen_modelo(meta)}
    return resultado, usadas
