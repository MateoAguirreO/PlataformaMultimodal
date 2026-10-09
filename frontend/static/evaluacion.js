// Evaluacion de una muestra nueva (antes inline en index.html: se movio aca para que la
// politica CSP pueda prohibir scripts inline). Usa api() y esc() de auth.js.
const $ = (sel) => document.querySelector(sel);
const state = { audio: null, video: null, wav: null, voz: null };   // voz: embedding del WAV, ya calculado
const COLOR_MODALIDAD = { voz: "var(--series-audio)", rostro: "var(--series-video)" };

// Mensaje de error de FastAPI: texto, {codigo, mensaje} o lista de errores de validacion.
const detalle = (d) => (typeof d === "string" ? d
  : Array.isArray(d) ? d.map((e) => e.msg).join(" · ") : (d && d.mensaje) || "error desconocido");

function cuerpo() {
  return JSON.stringify({ audio_segments: state.audio, video_features: state.video,
                          voz_embedding: state.voz ? state.voz.embedding : undefined });
}

// El embedding de voz se calcula una vez por WAV (en el backend) y se reutiliza.
async function asegurarVoz(btn) {
  if (!state.wav || state.voz) return;
  btn.textContent = "Procesando el audio (wav2vec2)...";
  const fd = new FormData();
  fd.append("audio", state.wav);
  const resp = await api("/voz/embedding", { method: "POST", body: fd });
  const data = await resp.json();
  if (!resp.ok) throw new Error(`audio del participante: ${detalle(data.detail)}`);
  state.voz = data;
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.length);
  if (!lines.length) return [];
  const headers = lines[0].split(",").map(h => h.trim());
  return lines.slice(1).map(line => {
    const cells = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      const v = parseFloat(cells[i]);
      if (!Number.isNaN(v)) row[h] = v;
    });
    return row;
  });
}

async function readFile(file) { return await file.text(); }

$("#audio-file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  state.audio = parseCSV(await readFile(f));
  checkReady();
});

$("#video-file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const text = await readFile(f);
  try {
    state.video = JSON.parse(text);
  } catch {
    const rows = parseCSV(text);
    state.video = rows[0] || null;
  }
  checkReady();
});

$("#wav-file").addEventListener("change", (e) => {
  state.wav = e.target.files[0] || null;
  state.voz = null;
  $("#quitar-wav").hidden = !state.wav;
});

$("#quitar-wav").addEventListener("click", () => {
  const input = $("#wav-file");
  input.value = "";
  input.dispatchEvent(new Event("change"));
});

function checkReady() {
  $("#btn-evaluar").disabled = !(state.audio && state.audio.length && state.video);
}

function showError(msg) {
  const el = $("#error");
  el.textContent = msg;
  el.style.display = "block";
  $("#resultados").style.display = "none";
}

function hideError() { $("#error").style.display = "none"; }

function ejeCard(dx, r) {
  const riesgo = r.clase === "riesgo";
  const pct = Math.round(r.p_riesgo * 100);
  const color = riesgo ? "var(--critical)" : "var(--good)";
  return `
    <div class="card eje-card">
      <div class="eje-head">
        <span class="eje-nombre">${dx}</span>
        <span class="badge ${riesgo ? "riesgo" : "sin-riesgo"}">
          ${riesgo ? "⚠ riesgo" : "✓ sin riesgo"}
        </span>
      </div>
      <div class="hero">
        <span class="num" style="color:${color}">${pct}%</span>
        <span class="lbl">probabilidad de riesgo (soft vote de las ramas)</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%; background:${color}"></div></div>

      <div class="legend">
        <span><span class="dot" style="background:var(--series-audio)"></span>voz</span>
        <span><span class="dot" style="background:var(--series-video)"></span>rostro</span>
      </div>
      <div class="sub-bars">${r.ramas.map(ramaRow).join("")}</div>
      ${avisoParcial(r)}
      <div class="modelo-line">
        ${r.ramas.map((x) => esc(x.nombre)).join(" + ")} · fusión soft vote ·
        AUC en validación cruzada ${r.modelo.auc_combinacion_cv.toFixed(2)} (optimista: la mejor de 127 combinaciones;
        con selección anidada ${r.modelo.auc_seleccion_anidada.toFixed(2)}) · etiqueta ${esc(r.modelo.etiqueta)}
      </div>
      <button class="explain-btn" data-eje="${dx}">Ver explicación (SHAP + verificación IA jr/senior)</button>
      <div class="explain-body" id="explain-${dx}" hidden></div>
    </div>`;
}

function ramaRow(rama) {
  const color = COLOR_MODALIDAD[rama.modalidad] || "var(--text-muted)";
  const etiqueta = `<span class="lbl"><span class="dot" style="background:${color}"></span>${esc(rama.modalidad)}</span>`;
  if (!rama.disponible) {
    return `<div class="sub-bar-row">${etiqueta}<span class="rama-nd">no disponible: ${esc(rama.motivo)}</span></div>`;
  }
  const pct = Math.round(rama.prob * 100);
  return `
    <div class="sub-bar-row" title="${esc(rama.nombre)}">
      ${etiqueta}
      <div class="sub-track"><div class="sub-fill" style="width:${pct}%; background:${color}"></div></div>
      <span class="val">${pct}%</span>
    </div>`;
}

// Si falta una rama, la predicción la hace el subconjunto disponible: otro modelo, con
// otro rendimiento (p. ej. depresión sin audio = solo rostro, que marca a más personas).
function avisoParcial(r) {
  if (!r.parcial || !r.rendimiento) return "";
  const u = r.rendimiento.umbral;
  const usadas = r.ramas.filter((x) => x.disponible).map((x) => esc(x.modalidad)).join(" + ");
  return `<div class="aviso-parcial"><strong>Evaluación parcial: solo ${usadas}.</strong> Es otro modelo:
    AUC en validación cruzada ${r.rendimiento.auc_cv.toFixed(2)}; con el umbral ${u.valor}, sensibilidad
    ${u.sensibilidad.toFixed(2)} y especificidad ${u.especificidad.toFixed(2)}.</div>`;
}

function notaExplicacion(shap) {
  const signo = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`;
  let html = `<div class="hint">p_riesgo ${shap.p_riesgo.toFixed(3)} = base ${shap.base.toFixed(3)}
    ${signo(shap.suma_contribuciones)} de contribuciones. Cada barra es el SHAP de la feature dividido
    por el número de ramas de la fusión.</div>`;
  for (const b of shap.top_features.filter((f) => f.proxy_egemaps && f.proxy_egemaps.length)) {
    html += `<div class="hint">${esc(b.feature)}: no es interpretable por dimensión; su barra es el aporte del
      bloque completo. En el entrenamiento, su predicción se asocia con
      ${b.proxy_egemaps.map((x) => `${esc(x.feature)} (ρ ${x.rho >= 0 ? "+" : "−"}${Math.abs(x.rho).toFixed(2)})`).join(", ")}.</div>`;
  }
  const m = shap.modelo;
  const u = shap.parcial && shap.rendimiento ? shap.rendimiento.umbral : m.umbral;
  const umbral = u ? `<li>Con el umbral ${u.valor}, en validación cruzada${shap.parcial ? " (solo las ramas usadas)" : ""}:
    sensibilidad ${u.sensibilidad.toFixed(2)}, especificidad ${u.especificidad.toFixed(2)}.</li>` : "";
  html += `<details class="limites"><summary>Limitaciones del modelo</summary><ul>${umbral}
    ${m.advertencias.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></details>`;
  return html;
}

function shapBars(top_features) {
  const maxAbs = Math.max(...top_features.map(f => Math.abs(f.shap_value)), 1e-9);
  return top_features.map(f => {
    const pct = Math.min(100, Math.abs(f.shap_value) / maxAbs * 100);
    const pos = f.shap_value >= 0;
    const dotColor = COLOR_MODALIDAD[f.modalidad] || "var(--text-muted)";
    return `
      <div class="shap-row">
        <span class="shap-lbl">
          <span class="dot" style="background:${dotColor}"></span>
          <span class="fname">${esc(f.feature)}</span>
          <span class="fam-tag">${esc(f.familia)}</span>
        </span>
        <div class="shap-track">
          <div class="shap-mid"></div>
          <div class="shap-fill ${pos ? "pos" : "neg"}" style="width:${pct / 2}%"></div>
        </div>
        <span class="shap-val">${pos ? "+" : ""}${f.shap_value.toFixed(3)}</span>
      </div>`;
  }).join("");
}

function reflexionTimeline(r) {
  const rounds = r.historial.map(h => {
    const s = h.auditoria_senior;
    const ok = s.aprobado;
    const errores = s.errores_numericos || [];
    const noSost = s.lecturas_no_sostenidas || [];
    return `
      <div class="round-card">
        <div class="round-head">
          <span>Ronda ${h.ronda} — borrador del jr</span>
          <span class="verdict-badge ${ok ? "ok" : "bad"}">${ok ? "✓ aprobado por el senior" : "✗ rechazado"}</span>
        </div>
        <div class="round-jr">${esc(h.borrador_jr.resumen)}</div>
        ${errores.length ? `<div class="warn-item">⚠ errores numéricos: ${errores.map(esc).join(" · ")}</div>` : ""}
        ${noSost.length ? `<div class="warn-item">⚠ lecturas sin respaldo clínico: ${noSost.map(esc).join(" · ")}</div>` : ""}
        ${!ok && s.feedback_para_junior ? `<div class="round-feedback">feedback del senior al jr: ${esc(s.feedback_para_junior)}</div>` : ""}
      </div>`;
  }).join("");
  const vf = r.version_final || {};
  return `
    <h3>Proceso de verificación jr/senior (${r.rondas} ronda${r.rondas > 1 ? "s" : ""}${r.estancado ? " — sin convergencia" : ""})</h3>
    ${rounds}
    <div class="final-block">
      <div class="final-head">${r.aprobado_final ? "✓ versión final aprobada" : "último intento (no aprobado)"}</div>
      <p style="margin:0 0 4px">${esc(vf.resumen)}</p>
      <ul class="lectura-list">
        ${(vf.lectura_clinica || []).map(l => `
          <li>
            <strong>${esc(l.feature)}</strong>: ${esc(l.conclusion_simple || l.lectura)}
            <div class="lectura-tecnica">${esc(l.lectura)}${l.con_prior ? "" : " <em>(sin respaldo clínico citado)</em>"}</div>
          </li>`
        ).join("")}
      </ul>
    </div>`;
}

function renderExplain(dx, data) {
  const el = document.getElementById(`explain-${dx}`);
  let html = `<h3>Features que más influyeron en este caso</h3>${shapBars(data.shap.top_features)}`;
  html += notaExplicacion(data.shap);
  html += data.reflexion
    ? reflexionTimeline(data.reflexion)
    : `<div class="hint" style="margin-top:14px">${esc(data.reflexion_nota || "verificación jr/senior no disponible")}</div>`;
  el.innerHTML = html;
  el.hidden = false;
}

document.addEventListener("click", async (e) => {
  const btn = e.target.closest(".explain-btn");
  if (!btn) return;
  const dx = btn.dataset.eje;
  const targetEl = document.getElementById(`explain-${dx}`);
  btn.disabled = true;
  try {
    await asegurarVoz(btn);
    btn.textContent = "Analizando (SHAP + verificación jr/senior)...";
    const resp = await api(`/explain?eje=${encodeURIComponent(dx)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: cuerpo(),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(detalle(data.detail));
    renderExplain(dx, data);
    btn.remove();
  } catch (err) {
    targetEl.innerHTML = `<div class="hint" style="color:var(--critical)">Error: ${esc(err.message)}</div>`;
    targetEl.hidden = false;
    btn.disabled = false;
    btn.textContent = "Ver explicación (SHAP + verificación IA jr/senior)";
  }
});

$("#btn-evaluar").addEventListener("click", async () => {
  hideError();
  const btn = $("#btn-evaluar");
  btn.disabled = true;
  try {
    await asegurarVoz(btn);
    btn.textContent = "Evaluando...";
    const resp = await api("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: cuerpo(),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(detalle(data.detail));
    const cont = $("#resultados");
    cont.innerHTML = Object.entries(data).map(([dx, r]) => ejeCard(dx, r)).join("");
    cont.style.display = "block";
    $("#informe-wrap").hidden = false;
  } catch (err) {
    showError("Error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Evaluar";
  }
});

$("#btn-informe").addEventListener("click", async () => {
  hideError();
  const btn = $("#btn-informe");
  btn.disabled = true;
  const original = btn.textContent;
  try {
    await asegurarVoz(btn);
    btn.textContent = "Generando informe (SHAP + verificación jr/senior de los 2 ejes)...";
    const resp = await api("/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: cuerpo(),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail ? detalle(data.detail) : `error ${resp.status}`);
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "informe_riesgo.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    showError("Error generando el informe: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
});
