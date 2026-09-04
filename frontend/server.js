// Frontend Node/Express: sirve la interfaz estatica y hace de proxy de
// /api/* hacia el backend Python (FastAPI) -- BFF simple, sin CORS, un solo
// puerto expuesto afuera (patron estandar para microservicios en contenedores).
import express from "express";
import { createProxyMiddleware } from "http-proxy-middleware";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
const PORT = process.env.PORT || 3000;

const app = express();

// /api/predict -> BACKEND_URL/predict ; /api/health -> BACKEND_URL/health ; etc.
app.use(
  "/api",
  createProxyMiddleware({
    target: BACKEND_URL,
    changeOrigin: true,
    pathRewrite: { "^/api": "" },
  })
);

app.use(express.static(path.join(__dirname, "static")));

app.listen(PORT, () => {
  console.log(`[frontend] escuchando en :${PORT}, proxy /api -> ${BACKEND_URL}`);
});
