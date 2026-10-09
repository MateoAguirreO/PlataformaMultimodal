// Frontend Node/Express (BFF): sirve la interfaz, protege las paginas internas (exigen
// sesion, que verifica el backend) y hace de proxy de /api/* hacia el backend Python
// (FastAPI) -- sin CORS, un solo puerto expuesto afuera; el backend no se publica.
import express from "express";
import { createProxyMiddleware } from "http-proxy-middleware";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.join(__dirname, "static");
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
const PORT = process.env.PORT || 3000;
const HTTPS = (process.env.COOKIE_SECURE || "false").toLowerCase() === "true";

const app = express();
app.disable("x-powered-by");

// Detras de un proxy inverso con HTTPS (Caddy, nginx): TRUST_PROXY=1 (saltos) o la IP/subred
// del proxy, para que req.ip sea la del cliente real. Sin proxy, dejarlo vacio: si no, un
// cliente podria inventarse la IP con X-Forwarded-For.
const trustProxy = process.env.TRUST_PROXY;
if (trustProxy && trustProxy !== "false") {
  app.set("trust proxy", /^\d+$/.test(trustProxy) ? Number(trustProxy) : trustProxy);
}

// --- Encabezados de seguridad (equivalente minimo de helmet) ----------------------------
const CSP = [
  "default-src 'self'",
  "script-src 'self'",                  // nada de scripts inline ni de terceros
  "style-src 'self' 'unsafe-inline'",   // estilos inline de las paginas (no ejecutan codigo)
  "img-src 'self' data:",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join("; ");

app.use((req, res, next) => {
  res.set({
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    // la interfaz sube archivos; no graba. Si algun dia graba en el navegador: camera=(self)...
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
  });
  if (HTTPS) res.set("Strict-Transport-Security", "max-age=31536000");
  next();
});

// --- API ---------------------------------------------------------------------------------
// La documentacion interactiva del backend (/docs, /openapi.json) no se expone hacia afuera.
app.use(["/api/docs", "/api/redoc", "/api/openapi.json"], (req, res) => res.status(404).end());

// /api/predict -> BACKEND_URL/predict ; /api/auth/login -> BACKEND_URL/auth/login ; etc.
app.use(
  "/api",
  (req, res, next) => {
    res.set("Cache-Control", "no-store");   // respuestas con datos clinicos: sin cache
    next();
  },
  createProxyMiddleware({
    target: BACKEND_URL,
    changeOrigin: true,
    pathRewrite: { "^/api": "" },
    on: {
      // La IP que ve el backend (auditoria, limite de intentos de login) la fija el BFF;
      // cualquier X-Forwarded-For que mande el cliente se descarta.
      proxyReq: (proxyReq, req) =>
        proxyReq.setHeader("x-forwarded-for", (req.ip || "").replace(/^::ffff:/, "")),   // IPv4 legible
      error: (err, req, res) => {
        console.error(`[frontend] proxy ${req.method} ${req.url}: ${err.code || err.message}`);
        if (typeof res.writeHead !== "function") return res.destroy?.();
        if (!res.headersSent) res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ detail: "El backend no responde. Intenta de nuevo en unos minutos." }));
      },
    },
  })
);

// --- Paginas ------------------------------------------------------------------------------
// Las internas solo se sirven con una sesion valida (el backend la verifica en /auth/me).
// Es una capa extra: los datos igual estan protegidos en cada endpoint de la API.
async function usuarioDeLaSesion(req) {
  const resp = await fetch(`${BACKEND_URL}/auth/me`, { headers: { cookie: req.headers.cookie || "" } });
  if (resp.status === 401) return null;
  if (!resp.ok) throw new Error(`/auth/me respondio ${resp.status}`);
  return resp.json();
}

function paginaProtegida(archivo, { soloAdmin = false, esCambioPassword = false } = {}) {
  return async (req, res) => {
    let u;
    try {
      u = await usuarioDeLaSesion(req);
    } catch (err) {
      console.error(`[frontend] no se pudo verificar la sesion: ${err.message}`);
      return res.status(503).type("text").send("Servicio no disponible. Intenta de nuevo en unos minutos.");
    }
    if (!u) return res.redirect("/login.html");
    if (u.debe_cambiar_password && !esCambioPassword) return res.redirect("/cambiar-password.html");
    if (soloAdmin && u.rol !== "admin") return res.redirect("/");
    res.set("Cache-Control", "no-store");
    res.sendFile(path.join(STATIC, archivo));
  };
}

app.get(["/", "/index.html"], paginaProtegida("index.html"));
app.get("/admin.html", paginaProtegida("admin.html", { soloAdmin: true }));
app.get("/cambiar-password.html", paginaProtegida("cambiar-password.html", { esCambioPassword: true }));

// Ningun otro .html se sirve directo (solo el login es publico); se decodifica la ruta para
// que variantes como /admin%2Ehtml no se salten la verificacion de sesion.
app.use((req, res, next) => {
  let ruta;
  try {
    ruta = decodeURIComponent(req.path).toLowerCase();
  } catch {
    return res.status(400).end();
  }
  if (/\.html?$/.test(ruta) && ruta !== "/login.html") return res.status(404).end();
  next();
});

app.use(express.static(STATIC, { index: false }));

app.listen(PORT, () => {
  console.log(`[frontend] escuchando en :${PORT}, proxy /api -> ${BACKEND_URL}`);
});
