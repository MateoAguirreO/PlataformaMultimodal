"""SHAP local para una muestra nueva, contra los modelos YA desplegados
(voz + rostro por separado, arquitectura soft_vote). No es el mismo modelo de
early-fusion que usa xai_shap_local.py en el repo de tesis (ese solo sirvio
para demostrar el loop jr/senior sobre 4 casos historicos) -- esto explica el
modelo que REALMENTE corre en produccion.

Rostro: 1 fila (el participante) -> SHAP exacto sobre las 66 features.
Voz: N filas (los segmentos de este participante) -> SHAP por segmento,
     promediado |SHAP| por feature entre segmentos (misma logica que el SHAP
     GLOBAL de voz del repo de tesis -- xai_shap.py::shap_audio -- aplicada
     aqui a los segmentos de un solo participante en vez de a todo el cohorte).

Usa el explainer exacto segun el tipo de modelo (Tree para XGBoost/RandomForest,
Linear para LogisticRegression) -- rapido, sin muestreo, igual que en el repo
de tesis. Requiere el background de train_final.py (<dx>_{audio,video}_bg.joblib)
solo para el caso lineal (LinearExplainer necesita un baseline).
"""
import numpy as np
import shap

FAMILIAS_AUDIO = {
    "F0/prosodia": ["F0semitone", "logRelF0"],
    "loudness/energia": ["loudness", "equivalentSoundLevel", "loudnessPeaksPerSec"],
    "espectral": ["spectralFlux", "alphaRatio", "hammarbergIndex", "slopeV", "slopeUV"],
    "MFCC": ["mfcc"],
    "calidad de voz": ["jitter", "shimmer", "HNRdBACF"],
    "formantes": ["F1", "F2", "F3"],
    "temporal/tasa": ["VoicedSegmentsPerSec", "MeanVoicedSegmentLength",
                      "StddevVoicedSegmentLength", "MeanUnvoicedSegmentLength",
                      "StddevUnvoicedSegmentLength"],
}
FAMILIAS_VIDEO = {
    "cara superior": ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU43"],
    "cara inferior": ["AU10", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20",
                      "AU23", "AU24", "AU25", "AU26", "AU28"],
    "pose de cabeza": ["Pitch", "Roll", "Yaw", "X_", "Y_", "Z_"],
    "emocion": ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"],
}


def _familia(feature, familias):
    for fam, patrones in familias.items():
        if any(p in feature for p in patrones):
            return fam
    return "otra"


def _indices_paso(step):
    if hasattr(step, "get_support"):
        return step.get_support(indices=True)
    if hasattr(step, "support_"):
        return np.asarray(step.support_)
    return None


def _nombres_finales(pipe, feats):
    idx = np.arange(len(feats))
    for _, step in pipe.steps[:-1]:
        sub = _indices_paso(step)
        if sub is not None:
            idx = idx[sub]
    return [feats[i] for i in idx]


def _shap_pos(explainer, Xt):
    sv = explainer(Xt) if callable(explainer) else explainer.shap_values(Xt)
    vals = sv.values if hasattr(sv, "values") else sv
    if isinstance(vals, list):
        vals = np.asarray(vals[1])
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    return vals


def _explicar_una_fila(pipe, X_raw, feats, bg_raw):
    """SHAP de UNA fila (X_raw: array (1, n_feats_crudas)) contra `pipe` completo.

    Usa TreeExplainer si el clasificador final es de arbol (exacto, sin
    background); si no, LinearExplainer sobre el espacio post-seleccion con
    `bg_raw` transformado como referencia.
    """
    ff = _nombres_finales(pipe, feats)
    Xt = pipe[:-1].transform(X_raw)
    clf = pipe.named_steps["clf"]
    try:
        expl = shap.TreeExplainer(clf)
        vals = _shap_pos(expl, Xt)
    except Exception:
        bg_t = pipe[:-1].transform(bg_raw)
        expl = shap.LinearExplainer(clf, bg_t)
        vals = _shap_pos(expl, Xt)
    return ff, vals  # vals: shape (n_filas, n_feats_seleccionadas)


def explicar(dx, modelos, audio_segments, video_features, top_k=15):
    """audio_segments: list[dict], video_features: dict -> payload compatible
    con xai_reflexion.py (mismo esquema que shap_local_<dx>_<caso>.json del
    repo de tesis: top_features con feature/modalidad/familia/valor/shap)."""
    m = modelos[dx]
    meta = m["meta"]
    feats_a, feats_v = meta["audio_features"], meta["video_features"]

    # --- rostro: 1 fila ---
    Xv = np.array([[video_features[f] for f in feats_v]])
    ff_v, sv_v = _explicar_una_fila(m["video"], Xv, feats_v, m["video_bg"])
    sv_v = sv_v[0]  # una sola fila

    # --- voz: N segmentos -> |SHAP| promedio por feature ---
    Xa = np.array([[s[f] for f in feats_a] for s in audio_segments])
    ff_a, sv_a = _explicar_una_fila(m["audio"], Xa, feats_a, m["audio_bg"])
    sv_a_mean_abs = np.abs(sv_a).mean(axis=0)
    # signo: promedio del SHAP real (no del abs) para saber si empuja a riesgo o no
    sv_a_mean = sv_a.mean(axis=0)

    p_video = float(m["video"].predict_proba(Xv)[0, 1])
    p_audio_segs = m["audio"].predict_proba(Xa)[:, 1]
    p_audio = float(np.mean(p_audio_segs))

    filas = []
    for f, v in zip(ff_v, sv_v):
        filas.append({"feature": f, "modalidad": "video", "familia": _familia(f, FAMILIAS_VIDEO),
                      "valor_feature": float(video_features[f]), "shap_value": round(float(v), 5)})
    for f, v in zip(ff_a, sv_a_mean):
        filas.append({"feature": f, "modalidad": "audio", "familia": _familia(f, FAMILIAS_AUDIO),
                      "valor_feature": float(np.mean([s[f] for s in audio_segments])),
                      "shap_value": round(float(v), 5)})
    filas.sort(key=lambda r: -abs(r["shap_value"]))

    p_riesgo = (p_audio + p_video) / 2.0
    return {
        "eje": dx,
        "p_riesgo": round(p_riesgo, 4),
        "prediccion_clase": "riesgo" if p_riesgo >= 0.5 else "sin riesgo",
        "score_audio": round(p_audio, 4),
        "score_video": round(p_video, 4),
        "n_segmentos_audio": len(audio_segments),
        "top_features": filas[:top_k],
    }
