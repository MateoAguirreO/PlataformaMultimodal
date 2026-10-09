"use strict";
const aviso = document.getElementById("aviso");
const cajaClave = document.getElementById("clave-temporal");
const json = (datos) => ({ headers: { "Content-Type": "application/json" }, body: JSON.stringify(datos) });

const ACCIONES = {
  login_ok: "Inicio de sesión",
  login_fallido: "Inicio de sesión fallido",
  login_rechazado_bloqueo: "Intento con cuenta bloqueada",
  cuenta_bloqueada: "Cuenta bloqueada",
  logout: "Cierre de sesión",
  cambio_password: "Cambio de contraseña",
  cambio_password_fallido: "Cambio de contraseña fallido",
  crear_usuario: "Usuario creado",
  actualizar_usuario: "Usuario modificado",
  restablecer_password: "Contraseña restablecida",
  admin_inicial_creado: "Administrador inicial creado",
  predict: "Evaluación",
  explain: "Explicación (SHAP + jr/senior)",
  report: "Informe PDF",
  predict_raw: "Evaluación desde audio/video",
  explain_raw: "Explicación desde audio/video",
  report_raw: "Informe desde audio/video",
  voz_embedding: "Embedding de voz (WAV)",
};
const ROLES = { admin: "Administrador", clinico: "Clínico" };

let yo = null;

function mostrarAviso(texto, tipo = "error") {
  aviso.textContent = texto;
  aviso.className = `aviso ${tipo}`;
  aviso.hidden = false;
  if (tipo === "ok") setTimeout(() => { aviso.hidden = true; }, 4000);
}

// La contraseña temporal se muestra una sola vez: el backend no la guarda en claro.
function mostrarClaveTemporal(email, clave, motivo) {
  cajaClave.innerHTML = `
    <strong>${esc(motivo)} ${esc(email)}</strong><br />
    <span class="clave-temporal">${esc(clave)}</span>
    ${navigator.clipboard ? `<button type="button" class="secundario" id="btn-copiar">Copiar</button>` : ""}
    <p class="hint">Contraseña temporal: se muestra solo esta vez. Entrégala por un canal privado;
      se pedirá cambiarla en el primer ingreso.</p>`;
  cajaClave.hidden = false;
  const copiar = document.getElementById("btn-copiar");
  if (copiar) {
    copiar.addEventListener("click", async () => {
      await navigator.clipboard.writeText(clave);
      copiar.textContent = "Copiada";
    });
  }
  cajaClave.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function estado(u) {
  const tags = [u.activo ? `<span class="etiqueta activo">activo</span>` : `<span class="etiqueta inactivo">inactivo</span>`];
  if (u.bloqueado) tags.push(`<span class="etiqueta alerta">bloqueado</span>`);
  if (u.debe_cambiar_password) tags.push(`<span class="etiqueta pendiente">contraseña temporal</span>`);
  return tags.join("");
}

function filaUsuario(u) {
  const esYo = yo && u.id === yo.id;
  const opciones = Object.entries(ROLES)
    .map(([v, t]) => `<option value="${v}" ${u.rol === v ? "selected" : ""}>${t}</option>`).join("");
  return `
    <tr>
      <td>${esc(u.nombre)}${esYo ? `<span class="yo">(tú)</span>` : ""}</td>
      <td>${esc(u.email)}</td>
      <td><select data-accion="rol" data-id="${u.id}" ${esYo ? "disabled title='No puedes cambiar tu propio rol'" : ""}>${opciones}</select></td>
      <td>${estado(u)}</td>
      <td class="fecha">${fechaLocal(u.ultimo_acceso)}</td>
      <td><div class="acciones">
        <button type="button" class="secundario" data-accion="reset" data-id="${u.id}">Restablecer contraseña</button>
        ${esYo ? "" : u.activo
          ? `<button type="button" class="peligro" data-accion="desactivar" data-id="${u.id}">Desactivar</button>`
          : `<button type="button" class="secundario" data-accion="activar" data-id="${u.id}">Activar</button>`}
      </div></td>
    </tr>`;
}

let usuarios = [];

async function cargarUsuarios() {
  const resp = await api("/admin/usuarios");
  if (!resp.ok) return mostrarAviso(await mensajeDeError(resp));
  usuarios = await resp.json();
  document.getElementById("tabla-usuarios").innerHTML = usuarios.map(filaUsuario).join("");
}

function detalle(d) {
  if (!d) return "";
  const valor = (v) => (v && typeof v === "object"
    ? Object.entries(v).map(([a, b]) => `${a} = ${b}`).join(", ")    // p. ej. cambios: activo = false
    : v);
  return Object.entries(d).map(([k, v]) => `${esc(k)}: ${esc(valor(v))}`).join(" · ");
}

async function cargarAuditoria() {
  const accion = document.getElementById("filtro-accion").value;
  const resp = await api(`/admin/auditoria?limite=200${accion ? `&accion=${encodeURIComponent(accion)}` : ""}`);
  if (!resp.ok) return mostrarAviso(await mensajeDeError(resp));
  const eventos = await resp.json();
  document.getElementById("tabla-auditoria").innerHTML = eventos.length
    ? eventos.map((e) => `
        <tr>
          <td class="fecha">${fechaLocal(e.fecha)}</td>
          <td>${esc(e.email || "—")}</td>
          <td>${esc(ACCIONES[e.accion] || e.accion)}</td>
          <td class="detalle">${detalle(e.detalle)}</td>
          <td class="ip">${esc(e.ip || "")}</td>
        </tr>`).join("")
    : `<tr><td colspan="5" class="vacio">Sin eventos.</td></tr>`;
}

async function actualizarUsuario(u, cambios, confirmacion) {
  if (confirmacion && !confirm(confirmacion)) return false;
  const resp = await api(`/admin/usuarios/${u.id}`, { method: "PATCH", ...json(cambios) });
  if (!resp.ok) {
    mostrarAviso(await mensajeDeError(resp));
    return false;
  }
  mostrarAviso(`Cambios guardados para ${u.email}.`, "ok");
  return true;
}

document.getElementById("form-nuevo").addEventListener("submit", async (e) => {
  e.preventDefault();
  const nombre = document.getElementById("n-nombre").value.trim();
  const email = document.getElementById("n-email").value.trim();
  const rol = document.getElementById("n-rol").value;
  if (!nombre || !email) return mostrarAviso("Escribe el nombre y el correo del nuevo usuario.");
  if (rol === "admin" && !confirm(`¿Crear a ${email} como ADMINISTRADOR? Podrá gestionar usuarios y ver toda la actividad.`)) return;
  const btn = document.getElementById("btn-crear");
  btn.disabled = true;
  try {
    const resp = await api("/admin/usuarios", { method: "POST", ...json({ nombre, email, rol }) });
    if (!resp.ok) return mostrarAviso(await mensajeDeError(resp));
    const nuevo = await resp.json();
    e.target.reset();
    aviso.hidden = true;
    mostrarClaveTemporal(nuevo.email, nuevo.password_temporal, "Usuario creado:");
    await Promise.all([cargarUsuarios(), cargarAuditoria()]);
  } finally {
    btn.disabled = false;
  }
});

document.getElementById("tabla-usuarios").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-accion]");
  if (!btn) return;
  const u = usuarios.find((x) => x.id === Number(btn.dataset.id));
  if (!u) return;
  btn.disabled = true;
  try {
    if (btn.dataset.accion === "reset") {
      if (!confirm(`¿Restablecer la contraseña de ${u.email}? Se le cerrarán las sesiones abiertas y tendrá que usar la contraseña temporal.`)) return;
      const resp = await api(`/admin/usuarios/${u.id}/reset-password`, { method: "POST" });
      if (!resp.ok) return mostrarAviso(await mensajeDeError(resp));
      const r = await resp.json();
      mostrarClaveTemporal(r.email, r.password_temporal, "Contraseña restablecida para");
    } else if (btn.dataset.accion === "desactivar") {
      await actualizarUsuario(u, { activo: false },
        `¿Desactivar a ${u.email}? No podrá ingresar y se cerrarán sus sesiones abiertas.`);
    } else if (btn.dataset.accion === "activar") {
      await actualizarUsuario(u, { activo: true });
    }
    await Promise.all([cargarUsuarios(), cargarAuditoria()]);
  } finally {
    btn.disabled = false;
  }
});

document.getElementById("tabla-usuarios").addEventListener("change", async (e) => {
  const sel = e.target.closest("select[data-accion=rol]");
  if (!sel) return;
  const u = usuarios.find((x) => x.id === Number(sel.dataset.id));
  const ok = await actualizarUsuario(u, { rol: sel.value },
    `¿Cambiar el rol de ${u.email} a ${ROLES[sel.value]}?`);
  if (!ok) sel.value = u.rol;
  await Promise.all([cargarUsuarios(), cargarAuditoria()]);
});

const filtro = document.getElementById("filtro-accion");
filtro.innerHTML += Object.entries(ACCIONES).map(([v, t]) => `<option value="${v}">${esc(t)}</option>`).join("");
filtro.addEventListener("change", cargarAuditoria);
document.getElementById("btn-actualizar").addEventListener("click", () => Promise.all([cargarUsuarios(), cargarAuditoria()]));

(async () => {
  yo = await sesionLista;
  await Promise.all([cargarUsuarios(), cargarAuditoria()]);
})().catch((err) => mostrarAviso(err.message));
