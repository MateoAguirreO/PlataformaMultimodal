"""SHAP local de una muestra nueva contra los modelos desplegados (ver ramas.py).

Es la misma explicación del análisis por muestra de la tesis
(Psiquiatria/Multimodal_XAI/xai_por_muestra.py), aplicada a una muestra nueva:
- Ramas interpretables (rostro AU, eGeMAPS): SHAP Permutation sobre el pipeline completo, en
  probabilidad de riesgo. El fondo son centroides k-means del entrenamiento (no filas de
  participantes).
- Ramas de embedding (voz wav2vec2): el SHAP por dimensión no es interpretable. Se reporta el
  bloque entero con su contribución a la fusión, (p_rama − p media OOF) / n_ramas, igual que
  contribucion_modelos_<eje>.csv, junto con su asociación poblacional con eGeMAPS (proxy).
Como la fusión es un promedio, la contribución de cada feature a p_riesgo es su SHAP dividido
por el número de ramas usadas. Así, p_riesgo = base + suma de contribuciones, de forma exacta.
"""
import numpy as np
import shap

import ramas

SEMILLA = 2026

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


def _familia(feature, modalidad):
    for fam, patrones in (FAMILIAS_VIDEO if modalidad == "rostro" else FAMILIAS_AUDIO).items():
        if any(p in feature for p in patrones):
            return fam
    return "otra"


def _shap(bundle, x):
    """SHAP Permutation (2 permutaciones antitéticas) de una fila -> (valores, valor base)."""
    fondo = bundle["fondo_shap"]
    ex = shap.explainers.Permutation(lambda X: bundle["pipeline"].predict_proba(X)[:, 1],
                                     shap.maskers.Independent(fondo, max_samples=len(fondo)), seed=SEMILLA)
    x = np.where(np.isnan(x), bundle["medianas"], x)
    r = ex(x[None, :], max_evals=2 * (2 * x.size + 1), silent=True)
    return r.values[0], float(r.base_values[0])


def explicar(eje, modelos, muestra, top_k=15):
    """-> payload para la interfaz, reflexion.py y report_pdf.py."""
    m = modelos[eje]
    meta = m["meta"]
    pred, usadas = ramas.evaluar(m, muestra)
    n = len(usadas)
    features, bloques, base = [], [], 0.0
    for b, (x, p) in usadas.items():
        r, bundle = meta["ramas"][b], m["ramas"][b]
        if r["interpretable"]:
            sv, base_rama = _shap(bundle, x)
            base += base_rama / n
            features += [{"feature": f, "rama": b, "modalidad": r["modalidad"], "familia": _familia(f, r["modalidad"]),
                          "valor_feature": round(float(v_x), 5), "shap_value": round(float(v) / n, 5)}
                         for f, v, v_x in zip(r["features"], sv, x) if v != 0]   # 0 = no la usa el modelo
        else:
            base += r["prob_media_oof"] / n
            bloques.append({"feature": r["nombre"], "rama": b, "modalidad": r["modalidad"],
                            "familia": "embedding (no interpretable por dimensión)", "valor_feature": None,
                            "shap_value": round((p - r["prob_media_oof"]) / n, 5),
                            "proxy_egemaps": r.get("proxy_egemaps", [])})
    features.sort(key=lambda d: -abs(d["shap_value"]))
    total = sum(d["shap_value"] for d in features + bloques)
    return {
        "eje": eje,
        "p_riesgo": pred["p_riesgo"],
        "prediccion_clase": pred["clase"],
        "ramas": pred["ramas"],
        "parcial": pred["parcial"],
        "rendimiento": pred["rendimiento"],
        "modelo": pred["modelo"],
        "base": round(base, 4),
        "suma_contribuciones": round(total, 4),
        # los bloques de embedding siempre van: son una modalidad entera
        "top_features": sorted(features[:top_k] + bloques, key=lambda d: -abs(d["shap_value"])),
    }
