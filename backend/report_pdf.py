"""Genera el informe en PDF de una muestra evaluada: prediccion, grafico SHAP
y proceso completo de verificacion jr/senior, por eje (ansiedad, depresion).

Pensado para uso academico (tesis de maestria): el informe expone el proceso
interno de la IA (borrador del jr, veredicto del senior, prior clinico citado)
en vez de esconderlo como caja negra -- ver README, seccion "por que se
expone el proceso interno".

Usa reportlab (layout de texto/tablas) + matplotlib (grafico de barras SHAP,
mismo lenguaje visual que el resto de la tesis) -- ambos pip puros, sin
dependencias de sistema, para que el Dockerfile no necesite libs nativas.
"""
import io
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                 Spacer, Table, TableStyle)

COLOR_AUDIO = "#2a78d6"
COLOR_VIDEO = "#eb6834"
COLOR_RIESGO = "#d03b3b"
COLOR_SIN_RIESGO = "#0ca30c"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13)
MUTED = ParagraphStyle("Muted", parent=BODY, textColor=colors.HexColor("#6b6a66"), fontSize=8.5)


def _shap_chart(top_features, titulo):
    top = top_features[:12][::-1]  # orden ascendente para barh
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(top) + 0.6))
    vals = [f["shap_value"] for f in top]
    colores = [COLOR_RIESGO if v >= 0 else COLOR_SIN_RIESGO for v in vals]
    ax.barh(range(len(top)), vals, color=colores, height=0.6)
    labels = [f"{f['feature']}  ({f['modalidad']})" for f in top]
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_title(titulo, fontsize=9.5)
    ax.tick_params(axis="x", labelsize=7.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf


CELL = ParagraphStyle("Cell", fontSize=8, leading=10)


def _tabla_lectura(lectura_clinica):
    if not lectura_clinica:
        return None
    # Paragraph, no strings planas -- una Table de reportlab NO hace wrap de
    # texto dentro de una celda si el contenido es un string, se desborda
    # sobre la columna siguiente en vez de partirse en varias lineas.
    data = [["Feature", "Lectura clinica", "Prior"]]
    for l in lectura_clinica:
        data.append([Paragraph(l.get("feature", ""), CELL),
                     Paragraph(l.get("lectura", ""), CELL),
                     "si" if l.get("con_prior") else "no"])
    t = Table(data, colWidths=[4.2 * cm, 9.5 * cm, 1.5 * cm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
    ]))
    return t


def _seccion_eje(dx, datos, story):
    shap = datos["shap"]
    refl = datos.get("reflexion")
    riesgo = shap["prediccion_clase"] == "riesgo"

    story.append(Paragraph(dx.capitalize(), H2))
    color = COLOR_RIESGO if riesgo else COLOR_SIN_RIESGO
    story.append(Paragraph(
        f'<font color="{color}"><b>{shap["prediccion_clase"].upper()}</b></font> — '
        f'p_riesgo = {shap["p_riesgo"]:.3f}  '
        f'(voz={shap["score_audio"]:.3f}, rostro={shap["score_video"]:.3f}, '
        f'{shap["n_segmentos_audio"]} segmentos de audio)', BODY))
    story.append(Spacer(1, 6))

    img_buf = _shap_chart(shap["top_features"], f"{dx} — features con mayor |SHAP|")
    iw, ih = ImageReader(img_buf).getSize()
    img_buf.seek(0)
    ancho = 15.5 * cm
    story.append(Image(img_buf, width=ancho, height=ancho * ih / iw))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Verificacion jr/senior (IA agentica, patron Reflexion — Shinn et al. 2023)", H3))
    if refl is None:
        story.append(Paragraph(
            datos.get("reflexion_nota", "verificacion jr/senior no disponible (falta GEMINI_API_KEY)"),
            MUTED))
    else:
        estado = "aprobada" if refl["aprobado_final"] else "no aprobada (mejor intento)"
        story.append(Paragraph(
            f"{refl['rondas']} ronda(s) · version final <b>{estado}</b>"
            f"{' · sin convergencia' if refl.get('estancado') else ''}", MUTED))
        story.append(Spacer(1, 4))
        vf = refl.get("version_final", {})
        if vf.get("resumen"):
            story.append(Paragraph(vf["resumen"], BODY))
        tabla = _tabla_lectura(vf.get("lectura_clinica"))
        if tabla:
            story.append(Spacer(1, 4))
            story.append(tabla)

        for h in refl["historial"]:
            s = h["auditoria_senior"]
            ok = s.get("aprobado")
            veredicto = ('<font color="#0ca30c">aprobado</font>' if ok
                         else '<font color="#d03b3b">rechazado</font>')
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                f"Ronda {h['ronda']} — {veredicto}",
                ParagraphStyle("RoundHead", parent=BODY, fontName="Helvetica-Bold")))
            story.append(Paragraph(f"jr: {h['borrador_jr'].get('resumen','')}", BODY))
            for err in s.get("errores_numericos", []):
                story.append(Paragraph(f"⚠ error numerico: {err}", MUTED))
            for ns in s.get("lecturas_no_sostenidas", []):
                story.append(Paragraph(f"⚠ no sostenida por el prior: {ns}", MUTED))


def generar(secciones: dict, meta: dict | None = None) -> bytes:
    """secciones: {dx: {"shap": ..., "reflexion": ... | None, "reflexion_nota": ...}}"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER,
                             topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    story = []
    story.append(Paragraph("Informe de evaluación — riesgo ansiedad / depresión", H1))
    fecha = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(
        f"Generado {fecha} · modelos voz+rostro (fusión soft_vote) · "
        f"uso académico, no es un diagnóstico clínico.", MUTED))
    story.append(Spacer(1, 10))

    for i, (dx, datos) in enumerate(secciones.items()):
        if i > 0:
            story.append(PageBreak())
        _seccion_eje(dx, datos, story)

    doc.build(story)
    return buf.getvalue()
