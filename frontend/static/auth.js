// Cliente común de todas las páginas: llamadas a la API con el encabezado anti-CSRF,
// manejo de sesión expirada y barra de sesión. La sesión vive en una cookie httpOnly que
// el navegador manda solo; este código nunca ve ni guarda el token.
"use strict";

// Como fetch(), pero con la ruta relativa a /api y el encabezado X-Requested-With que el
// backend exige en toda petición que modifica datos. Si la sesión expiró, o si falta
// cambiar la contraseña, lleva a la página que corresponde.
async function api(ruta, opciones = {}) {
  const headers = new Headers(opciones.headers || {});
  headers.set("X-Requested-With", "fetch");
  const resp = await fetch("/api" + ruta, { ...opciones, headers, credentials: "same-origin" });
  if (resp.status === 401 && !ruta.startsWith("/auth/login") && location.pathname !== "/login.html") {
    location.assign("/login.html?expirada=1");
    throw new Error("Tu sesión expiró. Inicia sesión de nuevo.");
  }
  if (resp.status === 403 && location.pathname !== "/cambiar-password.html") {
    const datos = await resp.clone().json().catch(() => ({}));
    if (datos.detail && datos.detail.codigo === "cambio_password_requerido") {
      location.assign("/cambiar-password.html");
      throw new Error(datos.detail.mensaje);
    }
  }
  return resp;
}

// Mensaje legible de una respuesta de error del backend (FastAPI).
async function mensajeDeError(resp) {
  const datos = await resp.json().catch(() => ({}));
  const d = datos.detail;
  if (typeof d === "string") return d;
  if (d && d.mensaje) return d.mensaje;
  if (Array.isArray(d)) return d.map((e) => e.msg).join(" · ");   // errores de validación
  return `Error ${resp.status}`;
}

// Escapa texto antes de insertarlo con innerHTML (nombres, correos, texto generado por IA).
function esc(valor) {
  return String(valor ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function fechaLocal(iso) {
  return iso ? new Date(iso).toLocaleString("es-CO", { dateStyle: "short", timeStyle: "short" }) : "—";
}

async function cerrarSesion() {
  try {
    await api("/auth/logout", { method: "POST" });
  } finally {
    location.assign("/login.html?salida=1");
  }
}

// Pinta la barra de sesión en <nav id="barra-sesion"> y devuelve el usuario actual.
async function iniciarBarraSesion() {
  const barra = document.getElementById("barra-sesion");
  if (!barra) return null;
  const resp = await api("/auth/me");
  if (!resp.ok) return null;
  const u = await resp.json();
  barra.innerHTML = `
    <a href="/" class="marca">Plataforma multimodal</a>
    <span>${esc(u.nombre)}<span class="rol">${u.rol === "admin" ? "administrador" : "clínico"}</span></span>
    ${u.rol === "admin" ? `<a href="/admin.html">Administración</a>` : ""}
    <a href="/cambiar-password.html">Cambiar contraseña</a>
    <button type="button" class="salir">Salir</button>`;
  barra.querySelector(".salir").addEventListener("click", cerrarSesion);
  barra.hidden = false;
  return u;
}

// Las páginas pueden esperar al usuario actual con: const yo = await sesionLista;
const sesionLista = iniciarBarraSesion().catch(() => null);
