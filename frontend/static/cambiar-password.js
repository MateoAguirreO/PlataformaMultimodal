"use strict";
const aviso = document.getElementById("aviso");

function mostrarAviso(texto, tipo) {
  aviso.textContent = texto;
  aviso.className = `aviso ${tipo}`;
  aviso.hidden = false;
}

sesionLista.then((u) => {
  if (u && u.debe_cambiar_password) {
    document.getElementById("motivo").textContent =
      "Antes de usar la plataforma debes reemplazar la contraseña temporal por una propia.";
  }
});

document.getElementById("form-cambio").addEventListener("submit", async (e) => {
  e.preventDefault();
  const actual = document.getElementById("actual").value;
  const nueva = document.getElementById("nueva").value;
  if (!actual || !nueva) return mostrarAviso("Completa los tres campos.", "error");
  if (nueva !== document.getElementById("confirmar").value) {
    return mostrarAviso("La nueva contraseña y su confirmación no coinciden.", "error");
  }
  const btn = document.getElementById("btn-guardar");
  btn.disabled = true;
  try {
    const resp = await api("/auth/cambiar-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actual, nueva }),
    });
    if (!resp.ok) {
      mostrarAviso(await mensajeDeError(resp), "error");
      btn.disabled = false;
      return;
    }
    e.target.reset();
    mostrarAviso("Contraseña actualizada. Entrando a la plataforma…", "ok");
    setTimeout(() => location.replace("/"), 1200);
  } catch (err) {
    mostrarAviso(err.message || "No se pudo contactar el servidor.", "error");
    btn.disabled = false;
  }
});
