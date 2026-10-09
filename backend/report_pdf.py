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
    labels = [f"{f['feature']}  ({f['modalidad']})" for f in top]   # rostro / voz
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
    data = [["Feature", "Lectura clinica", "Conclusion simple", "Respaldo clinico"]]
    for l in lectura_clinica:
        data.append([Paragraph(l.get("feature", ""), CELL),
                     Paragraph(l.get("lectura", ""), CELL),
                     Paragraph(l.get("conclusion_simple", ""), CELL),
                     "si" if l.get("con_prior") else "no"])
    t = Table(data, colWidths=[3.0 * cm, 6.0 * cm, 5.7 * cm, 2.2 * cm])
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
    ramas = "; ".join(f'{r["nombre"]} = {r["prob"]:.3f}' if r["disponible"]
                      else f'{r["nombre"]}: no disponible ({r.get("motivo", "")})' for r in shap["ramas"])
    story.append(Paragraph(
        f'<font color="{color}"><b>{shap["prediccion_clase"].upper()}</b></font> — '
        f'p_riesgo = {shap["p_riesgo"]:.3f} (promedio de: {ramas})', BODY))
    m = shap["modelo"]
    story.append(Paragraph(
        f'Modelo: {m["combinacion"]} (fusión soft vote) · AUC en validación cruzada {m["auc_combinacion_cv"]:.2f} '
        f'(optimista: mejor de 127 combinaciones) · con selección anidada {m["auc_seleccion_anidada"]:.2f} · '
        f'etiqueta {m["etiqueta"]}'
        + (f' · umbral {m["umbral"]["valor"]}: sensibilidad {m["umbral"]["sensibilidad"]:.2f}, '
           f'especificidad {m["umbral"]["especificidad"]:.2f}' if m.get("umbral") else ""), MUTED))
    rend = shap.get("rendimiento")
    if shap.get("parcial") and rend:
        usadas = " + ".join(r["nombre"] for r in shap["ramas"] if r["disponible"])
        story.append(Paragraph(
            f'<font color="#a86a06"><b>Evaluación parcial</b></font>: solo {usadas}. Es otro modelo: '
            f'AUC en validación cruzada {rend["auc_cv"]:.2f}; con el umbral {rend["umbral"]["valor"]}, '
            f'sensibilidad {rend["umbral"]["sensibilidad"]:.2f} y especificidad {rend["umbral"]["especificidad"]:.2f}.',
            BODY))
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

    story.append(Paragraph("Limitaciones del modelo", H3))
    for adv in shap["modelo"]["advertencias"]:
        story.append(Paragraph(f"• {adv}", MUTED))
    proxy = [b for b in shap["top_features"] if b.get("proxy_egemaps")]
    for b in proxy:
        asoc = ", ".join(f'{x["feature"]} (rho = {x["rho"]:+.2f})' for x in b["proxy_egemaps"])
        story.append(Paragraph(f"• {b['feature']}: no es interpretable por dimensión; en el entrenamiento "
                               f"su predicción se asocia con {asoc}.", MUTED))


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
        f"Generado {fecha} · un modelo por eje (rostro + voz, fusión soft vote) · "
        f"uso académico, no es un diagnóstico clínico.", MUTED))
    story.append(Spacer(1, 10))

    for i, (dx, datos) in enumerate(secciones.items()):
        if i > 0:
            story.append(PageBreak())
        _seccion_eje(dx, datos, story)

    doc.build(story)
    return buf.getvalue()
