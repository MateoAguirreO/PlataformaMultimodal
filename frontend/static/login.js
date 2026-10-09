"use strict";
const aviso = document.getElementById("aviso");

function mostrarAviso(texto, tipo) {
  aviso.textContent = texto;
  aviso.className = `aviso ${tipo}`;
  aviso.hidden = false;
}

const destino = (u) => (u.debe_cambiar_password ? "/cambiar-password.html" : "/");

// Con una sesión abierta no tiene sentido mostrar el login.
api("/auth/me")
  .then((r) => (r.ok ? r.json() : null))
  .then((u) => { if (u) location.replace(destino(u)); })
  .catch(() => {});

const params = new URLSearchParams(location.search);
if (params.has("expirada")) mostrarAviso("Tu sesión expiró o fue cerrada. Inicia sesión de nuevo.", "info");
if (params.has("salida")) mostrarAviso("Cerraste sesión.", "ok");

document.getElementById("form-login").addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = document.getElementById("email").value.trim();
  const campoPassword = document.getElementById("password");
  if (!email || !campoPassword.value) {
    mostrarAviso("Escribe tu correo y tu contraseña.", "error");
    return;
  }
  const btn = document.getElementById("btn-entrar");
  btn.disabled = true;
  btn.textContent = "Entrando…";
  try {
    const resp = await api("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password: campoPassword.value }),
    });
    if (!resp.ok) {
      mostrarAviso(await mensajeDeError(resp), "error");
      campoPassword.value = "";
      campoPassword.focus();
      return;
    }
    location.replace(destino(await resp.json()));
  } catch {
    mostrarAviso("No se pudo contactar el servidor. Intenta de nuevo.", "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Entrar";
  }
});
