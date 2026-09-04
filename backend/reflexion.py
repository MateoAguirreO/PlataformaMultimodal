"""Loop agentico de reflexion jr/senior sobre la explicacion SHAP de una
muestra nueva. Puerto de xai_reflexion.py (repo de tesis) como modulo
importable -- misma arquitectura (patron Evaluator-Optimizer, Shinn et al.
2023, NeurIPS -- ver pipeline_multimodal_xai/papers/ en el repo de tesis),
mismo prior clinico, mismo criterio de aprobacion.

Requiere GEMINI_API_KEY en el entorno. Si no esta configurada,
`reflexion_disponible()` devuelve False y el backend debe omitir esta etapa
en vez de fallar la respuesta completa (la prediccion + SHAP ya son utiles
sin esto).
"""
import json
import os
import re
import time

MODEL_JR = "gemini-3.1-flash-lite"
MODEL_SENIOR = "gemini-3.6-flash"
MAX_OUTPUT_TOKENS = 8000
N_RONDAS_MAX = 4

PRIOR_CLINICO = """
Prior clinico de la tesis (REPORTE_multimodal.md SS6). Usalo para chequear
plausibilidad; cualquier lectura fuera de esto es especulativa y debe marcarse:
- Voz, ansiedad: logRelF0-H1-H2 / logRelF0-H1-A3 (fonacion tensa vs soplada) +
  percentiles de loudness (energia vocal) -> tension laringea / control de intensidad.
- Voz, depresion: amplitudes de formantes relativas a F0 (FxamplitudeLogRelF0) +
  balance espectral (alphaRatio, hammarbergIndex) -> articulacion reducida, voz apagada.
- Rostro, ansiedad: variabilidad de AU15 (depresor comisura labial) y AU20
  (estirador labios) + AU01/AU04/AU05 (tension frontal/parpado) -> tension perioral y frontal.
- Rostro, depresion: prototipo 'anger' + AU01 (elevador interno de ceja) -> afecto
  negativo facial / ceno, patron clasico de tristeza en la literatura.
""".strip()

SYSTEM_JR = f"""Eres un analista JUNIOR de explicabilidad (XAI) para un modelo de
riesgo de ansiedad/depresion (fusion audio+video). Se te da el JSON de un caso
(SHAP local: prediccion, top features con su valor SHAP). Tu tarea:

1. CONSISTENCIA NUMERICA: revisa que el signo y la magnitud relativa de cada
   SHAP sean coherentes con la prediccion final (p_riesgo). Senala cualquier
   discrepancia grande entre score_audio y score_video (indica que las dos
   modalidades no coinciden para este caso).
2. LECTURA CLINICA: para las top features, da una lectura breve usando SOLO el
   prior clinico de abajo. Si una feature no esta en ese prior, dilo
   explicitamente como 'sin prior establecido' en vez de inventar una
   explicacion.

{PRIOR_CLINICO}

Responde EXCLUSIVAMENTE con un JSON (sin texto alrededor, sin markdown):
{{
  "chequeo_numerico": {{"consistente": bool, "nota": "..."}},
  "lectura_clinica": [{{"feature": "...", "lectura": "...", "con_prior": bool}}],
  "resumen": "..."
}}"""

SYSTEM_SENIOR = f"""Eres un analista SENIOR de explicabilidad (XAI), auditando el
borrador de un analista junior sobre un caso de SHAP local (modelo de riesgo de
ansiedad/depresion, fusion audio+video). Se te da el JSON original del caso y el
borrador del junior. Tu tarea:

1. Verifica cada afirmacion numerica del junior contra el JSON original (no le
   des el beneficio de la duda a su aritmetica).
2. Verifica que cada lectura clinica este realmente respaldada por el prior de
   abajo; marca como no sostenida cualquier lectura que el junior presente como
   firme sin estarlo.
3. Si se te da un historial de rondas previas, confirma que el junior
   REALMENTE corrigio lo que se le senalo.
4. Aprueba SOLO si no hay errores numericos y ninguna lectura clinica se
   presenta como mas firme de lo que el prior sostiene.

{PRIOR_CLINICO}

Responde EXCLUSIVAMENTE con un JSON (sin texto alrededor, sin markdown):
{{
  "aprobado": bool,
  "errores_numericos": ["..."],
  "lecturas_no_sostenidas": ["..."],
  "feedback_para_junior": "...",
  "version_final": {{
    "chequeo_numerico": {{"consistente": bool, "nota": "..."}},
    "lectura_clinica": [{{"feature": "...", "lectura": "...", "con_prior": bool}}],
    "resumen": "..."
  }}
}}"""


def reflexion_disponible():
    return bool(os.environ.get("GEMINI_API_KEY"))


def _extraer_json(texto):
    m = re.search(r"\{.*\}", texto, re.DOTALL)
    if not m:
        raise ValueError(f"no se encontro JSON en la respuesta:\n{texto[:500]}")
    return json.loads(m.group(0))


def _llamar(client, model, system, user, reintentos=4):
    from google.genai import errors, types
    config = types.GenerateContentConfig(
        system_instruction=system, max_output_tokens=MAX_OUTPUT_TOKENS, temperature=0.2)
    for intento in range(1, reintentos + 1):
        try:
            resp = client.models.generate_content(model=model, contents=user, config=config)
            return _extraer_json(resp.text)
        except errors.ServerError:
            if intento == reintentos:
                raise
            time.sleep(2 ** intento)


def _resumen_historial(historial):
    if not historial:
        return "(sin rondas previas)"
    bloques = []
    for h in historial:
        s = h["auditoria_senior"]
        bloques.append(
            f"-- Ronda {h['ronda']} --\n"
            f"  errores_numericos: {s.get('errores_numericos', [])}\n"
            f"  lecturas_no_sostenidas: {s.get('lecturas_no_sostenidas', [])}\n"
            f"  feedback: {s.get('feedback_para_junior', '')}")
    return "\n".join(bloques)


def _sin_progreso(historial):
    if len(historial) < 2:
        return False
    a, b = historial[-2]["auditoria_senior"], historial[-1]["auditoria_senior"]
    return (set(a.get("errores_numericos", [])) == set(b.get("errores_numericos", []))
            and set(a.get("lecturas_no_sostenidas", [])) == set(b.get("lecturas_no_sostenidas", [])))


def verificar(payload: dict) -> dict:
    """payload: el dict devuelto por xai.explicar(). Devuelve el resultado
    completo del loop jr/senior (mismo esquema que reflexion_<dx>_<caso>.json
    del repo de tesis)."""
    from google import genai
    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    caso_json = json.dumps(payload, ensure_ascii=False, indent=2)

    jr_out = _llamar(client, MODEL_JR, SYSTEM_JR, f"JSON del caso:\n{caso_json}")

    historial, senior_out, ronda, estancado = [], {}, 0, False
    for ronda in range(1, N_RONDAS_MAX + 1):
        senior_in = (f"JSON del caso:\n{caso_json}\n\n"
                     f"Historial de rondas previas:\n{_resumen_historial(historial)}\n\n"
                     f"Borrador del junior (ronda {ronda}):\n"
                     f"{json.dumps(jr_out, ensure_ascii=False, indent=2)}")
        senior_out = _llamar(client, MODEL_SENIOR, SYSTEM_SENIOR, senior_in)
        historial.append({"ronda": ronda, "borrador_jr": jr_out, "auditoria_senior": senior_out})

        if senior_out.get("aprobado") or ronda == N_RONDAS_MAX:
            break
        if _sin_progreso(historial):
            estancado = True
            break

        jr_in = (f"JSON del caso:\n{caso_json}\n\n"
                 f"Historial COMPLETO de feedback recibido hasta ahora:\n"
                 f"{_resumen_historial(historial)}\n\n"
                 f"Tu ultimo borrador fue RECHAZADO en la ronda {ronda}. Reescribe tu "
                 f"analisis corrigiendo TODOS los puntos senalados arriba.")
        jr_out = _llamar(client, MODEL_JR, SYSTEM_JR, jr_in)

    return {
        "rondas": ronda,
        "aprobado_final": senior_out.get("aprobado", False),
        "estancado": estancado,
        "version_final": senior_out.get("version_final", jr_out),
        "historial": historial,
    }
