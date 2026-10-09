"""Pruebas de los modelos por eje (ramas.py) y de su explicación (xai.py) con los modelos
reales de backend/modelos y las muestras de ejemplos/. No necesitan el modelo wav2vec2: el
embedding de voz se simula con un vector del tamaño correcto."""
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ramas  # noqa: E402
import xai  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]


def _filas(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [{k: float(v) for k, v in fila.items()} for fila in csv.DictReader(f)]


@pytest.fixture(scope="module")
def modelos():
    return ramas.cargar(RAIZ / "backend" / "modelos")


def muestra(tipo="positivo", voz=None):
    return SimpleNamespace(audio_segments=_filas(RAIZ / "ejemplos" / f"{tipo}_audio.csv"),
                           video_features=_filas(RAIZ / "ejemplos" / f"{tipo}_video.csv")[0],
                           voz_embedding=voz)


def embedding_tipico(modelos):
    """Mediana del entrenamiento: un embedding 'típico' del tamaño correcto."""
    return modelos["depresion"]["ramas"]["voz_microventanas"]["medianas"].tolist()


def test_cada_eje_tiene_su_combinacion(modelos):
    assert modelos["depresion"]["meta"]["combinacion"] == ["au", "voz_microventanas"]
    assert modelos["ansiedad"]["meta"]["combinacion"] == ["au", "egemaps"]
    for m in modelos.values():
        a = m["meta"]["auc"]
        assert a["seleccion_anidada"] < a["combinacion_cv"]       # la honesta es menor que la de la tabla


def test_sin_audio_del_participante_depresion_usa_solo_rostro(modelos):
    r, usadas = ramas.evaluar(modelos["depresion"], muestra())
    voz = next(x for x in r["ramas"] if x["rama"] == "voz_microventanas")
    assert not voz["disponible"] and "WAV" in voz["motivo"]
    assert list(usadas) == ["au"] and r["p_riesgo"] == round(usadas["au"][1], 4)
    meta = modelos["depresion"]["meta"]
    assert r["parcial"] and r["rendimiento"]["ramas"] == "au"
    assert r["rendimiento"]["auc_cv"] == meta["auc"]["por_rama"]["au"]          # otro modelo, otro AUC


def test_soft_vote_promedia_las_ramas_disponibles(modelos):
    for eje, m in (("depresion", muestra(voz=embedding_tipico(modelos))), ("ansiedad", muestra())):
        r, usadas = ramas.evaluar(modelos[eje], m)
        assert len(usadas) == 2 and all(x["disponible"] for x in r["ramas"])
        assert r["p_riesgo"] == round(float(np.mean([p for _, p in usadas.values()])), 4)
        assert r["clase"] == ("riesgo" if r["p_riesgo"] >= 0.5 else "sin riesgo")
        assert not r["parcial"] and r["rendimiento"]["auc_cv"] == modelos[eje]["meta"]["auc"]["combinacion_cv"]


def test_egemaps_es_la_media_de_los_segmentos(modelos):
    m = muestra()
    _, usadas = ramas.evaluar(modelos["ansiedad"], m)
    feats = modelos["ansiedad"]["meta"]["ramas"]["egemaps"]["features"]
    esperado = np.mean([[s[f] for f in feats] for s in m.audio_segments], axis=0)
    assert np.allclose(usadas["egemaps"][0], esperado)


def test_entradas_invalidas(modelos):
    with pytest.raises(ValueError, match="1024"):
        ramas.evaluar(modelos["depresion"], muestra(voz=[0.1] * 10))
    m = muestra()
    del m.video_features["AU01_mean"]
    with pytest.raises(ValueError, match="AU01_mean"):
        ramas.evaluar(modelos["ansiedad"], m)


@pytest.mark.parametrize("eje", ["depresion", "ansiedad"])
def test_explicacion_suma_exacta(modelos, eje):
    m = muestra(voz=embedding_tipico(modelos) if eje == "depresion" else None)
    e = xai.explicar(eje, modelos, m)
    assert abs(e["base"] + e["suma_contribuciones"] - e["p_riesgo"]) < 2e-3
    valores = [abs(f["shap_value"]) for f in e["top_features"]]
    assert valores == sorted(valores, reverse=True)
    assert {f["modalidad"] for f in e["top_features"]} <= {"rostro", "voz"}


def test_bloque_de_embedding_con_proxy(modelos):
    e = xai.explicar("depresion", modelos, muestra(voz=embedding_tipico(modelos)))
    bloques = [f for f in e["top_features"] if f["rama"] == "voz_microventanas"]
    assert len(bloques) == 1 and bloques[0]["proxy_egemaps"] and bloques[0]["valor_feature"] is None
