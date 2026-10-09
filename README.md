# PlataformaMultimodal

Evaluación de riesgo de ansiedad y depresión a partir de voz y rostro, con explicación por
muestra (SHAP + verificación jr/senior). Uso académico: no es un diagnóstico clínico.

## Modelos

Un modelo independiente por eje: la combinación de ramas con mejor AUC en el análisis por
muestra de la tesis, reentrenada con los 80 participantes (exportada el 2026-09-30 por
`Psiquiatria/Multimodal_XAI/exportar_plataforma.py`).

| Eje | Ramas (fusión soft vote) | AUC en CV | AUC con selección anidada | Umbral 0.5: sensibilidad / especificidad |
|---|---|---|---|---|
| Depresión | rostro (66 AU, py-feat) + voz (wav2vec2, micro-ventanas) | 0.70 | 0.56 | 0.45 / 0.89 |
| Ansiedad | rostro (66 AU, py-feat) + voz (88 eGeMAPS, media de segmentos) | 0.68 | 0.58 | 0.43 / 0.78 |

- El AUC en CV es el de la mejor de 127 combinaciones evaluadas en la misma validación
  cruzada, así que es optimista. La estimación honesta de "quedarse con la mejor" es la de
  selección anidada.
- La etiqueta es la columna "DX IA": tamizaje con PHQ-9 / GAD-7 y corte en síntomas leves (≥ 5).
  Coincide con las bandas del cuestionario en el 91–93% de los participantes.
- La voz de depresión (micro-ventanas de 0.19 s cada 2.5 s) no es robusta al punto de
  muestreo, y necesita el audio recortado (solo el participante) y sin ruido, como en el
  entrenamiento.

**Entradas.**

- Segmentos de audio (CSV de eGeMAPS por segmento) y rostro (CSV de 1 fila, o JSON):
  obligatorios.
- Audio del participante (WAV, recortado y sin ruido): opcional. Con él se calcula el
  embedding de voz de depresión con `wav2vec2-large-robust` (~1.2 GB, se descarga la primera
  vez). Sin él, depresión se evalúa solo con rostro, que es otro modelo (AUC en CV 0.61;
  umbral 0.5: sensibilidad 0.52, especificidad 0.69), y la interfaz lo marca como evaluación
  parcial.

**Actualizar los modelos.** Correr el exportador y copiar
`Multimodal_XAI/modelos_plataforma/*` sobre `backend/modelos/` (en Docker basta reiniciar).

## Requisitos

- Docker (recomendado), o Python 3.11 + Node 20 para correr sin Docker.
- Opcional: `GEMINI_API_KEY` para la verificación jr/senior (sin ella, todo lo demás funciona igual).

## Configurar

```bash
cp .env.example .env
# editar .env: GEMINI_API_KEY (opcional), correo del administrador inicial, etc.
```

## Correr con Docker

```bash
docker compose up --build
```

Abrir **http://localhost:3000** e iniciar sesión (ver [Usuarios y acceso](#usuarios-y-acceso)).

## Correr en local sin Docker

```bash
# terminal 1 — backend (solo en localhost: los usuarios entran por el frontend)
cd backend
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000

# terminal 2 — frontend
cd frontend
npm install
BACKEND_URL=http://localhost:8000 npm start
```

Abrir **http://localhost:3000**. Sin Docker, la base de usuarios queda en `backend/datos/plataforma.db`.

## Usuarios y acceso

**Primer arranque.** Si la base no tiene ningún administrador, el backend crea uno con
`ADMIN_EMAIL` (por defecto `admin@plataforma.local`). Si `ADMIN_PASSWORD` está vacío genera
una contraseña aleatoria y la muestra **una sola vez** en el log:

```bash
docker compose logs backend | grep -A1 "Administrador inicial"
```

Al primer ingreso pide cambiarla.

**Roles.**

| Rol | Puede |
|---|---|
| `clinico` | evaluar muestras, ver explicaciones y generar informes |
| `admin` | lo mismo + crear, desactivar y reactivar usuarios, restablecer contraseñas y ver la actividad (página *Administración*) |

**Cómo funciona.**

- Los usuarios los crea un administrador desde la web. La contraseña temporal se muestra una
  sola vez y hay que cambiarla en el primer ingreso. Los usuarios no se borran, se desactivan,
  para conservar la trazabilidad.
- Contraseñas: mínimo 10 caracteres, con letras y números, sin el usuario del correo. Se
  guardan con argon2id.
- Sesión en una cookie `httpOnly` + `SameSite=Strict`; en la base solo queda su hash. Expira
  tras 30 min sin uso o a las 8 h (`SESION_INACTIVIDAD_MIN`, `SESION_MAX_HORAS`). Cambiar o
  restablecer la contraseña, o desactivar al usuario, cierra sus sesiones abiertas.
- 5 intentos fallidos bloquean la cuenta 15 min. Además hay un límite de intentos por IP
  (`LOGIN_MAX_POR_IP`, 20 cada 15 min).
- Auditoría: inicios de sesión (también los fallidos), cambios de usuarios y cada evaluación
  (quién, cuándo, desde qué IP). No guarda datos clínicos ni contraseñas.
- Todos los endpoints de evaluación (`/predict`, `/explain`, `/report`, `*_raw`, `/schema`)
  exigen sesión; `/health` queda público.

**Recuperar el acceso** (p. ej. el único admin olvidó su contraseña):

```bash
docker compose exec backend python gestionar_usuarios.py listar
docker compose exec backend python gestionar_usuarios.py restablecer admin@plataforma.local
docker compose exec backend python gestionar_usuarios.py crear-admin persona@dominio.org "Nombre Apellido"
```

**Datos.** La base (SQLite) vive en el volumen `datos_plataforma`: sobrevive a rebuilds y a
`docker compose down`, pero `docker compose down -v` la borra. Respaldo en caliente:

```bash
docker compose exec backend python -c "import sqlite3; sqlite3.connect('/app/datos/plataforma.db').backup(sqlite3.connect('/app/datos/respaldo.db'))"
docker compose cp backend:/app/datos/respaldo.db ./respaldo_plataforma.db
```

Para otro motor (p. ej. PostgreSQL) basta con definir `DATABASE_URL`.

### Si se publica fuera de la máquina local

- Poner HTTPS delante (Caddy o nginx) y en `.env`: `COOKIE_SECURE=true` y `TRUST_PROXY=1`.
- Publicar el frontend solo hacia el proxy: `"127.0.0.1:3000:3000"` en `docker-compose.yml`.
- Respaldar el volumen `datos_plataforma` con regularidad.

## Probar rápido

Inicia sesión, sube los CSV de `ejemplos/` (`positivo_audio.csv`+`positivo_video.csv`, o
`negativo_audio.csv`+`negativo_video.csv`) y dale a Evaluar. Los ejemplos no traen WAV, así que depresión
sale solo con rostro; para probar su rama de voz, sube además un WAV con solo la voz del
participante.

## Pruebas

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest -q
```
