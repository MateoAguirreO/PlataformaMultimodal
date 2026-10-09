"""Autenticación, sesiones, roles y auditoría.

Diseño (patrón BFF, recomendado por OWASP para apps web con backend propio):
- Contraseñas con argon2id (argon2-cffi). Nunca se guardan ni se registran en claro.
- Sesiones del lado del servidor: el navegador recibe un token aleatorio de 256 bits en
  una cookie httpOnly + SameSite=Strict (+ Secure en producción); en la base solo queda su
  hash SHA-256. Expiran por inactividad y por duración máxima, y se pueden revocar
  (logout, cambio o restablecimiento de contraseña, desactivación del usuario).
- CSRF: además de SameSite=Strict, toda petición que modifica estado debe traer el
  encabezado `X-Requested-With: fetch`. Como el backend no habilita CORS, un sitio ajeno
  no puede enviarlo (defensa "custom request header" de OWASP).
- Fuerza bruta: bloqueo de la cuenta tras MAX_INTENTOS fallos seguidos, y límite de
  intentos de login por IP. Mensaje de error genérico (no revela si el usuario existe).
- Roles: admin (gestiona usuarios y ve la auditoría; también evalúa) y clinico (evalúa).
- Usuarios nuevos y contraseñas restablecidas exigen cambio de contraseña al ingresar.
"""
import hashlib
import json
import os
import secrets
import string
import time
from collections import defaultdict, deque
from datetime import timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, Response
from sqlalchemy import func, select, update

import db
from db import Evento, Sesion, Usuario, ahora

ROLES = ("admin", "clinico")
COOKIE = "pm_sesion"
CSRF_HEADER, CSRF_VALOR = "x-requested-with", "fetch"
MAX_INTENTOS = 5
BLOQUEO = timedelta(minutes=15)
VENTANA_IP_S = 15 * 60
MENSAJE_LOGIN = "Credenciales inválidas o cuenta bloqueada temporalmente."


def _cfg():
    """Configuración leída en cada uso (permite cambiarla por entorno en las pruebas)."""
    return {
        "cookie_secure": os.environ.get("COOKIE_SECURE", "false").lower() == "true",
        "inactividad": timedelta(minutes=int(os.environ.get("SESION_INACTIVIDAD_MIN", "30"))),
        "duracion_max": timedelta(hours=int(os.environ.get("SESION_MAX_HORAS", "8"))),
        # Intentos de login por IP cada 15 min. Con Docker Desktop todos los clientes pueden
        # llegar con la misma IP (la del NAT): en ese caso conviene subirlo.
        "login_por_ip": int(os.environ.get("LOGIN_MAX_POR_IP", "20")),
    }


# --------------------------------------------------------------------------- contraseñas
_ph = PasswordHasher()                       # argon2id con parámetros por defecto de la librería
_HASH_SEÑUELO = _ph.hash(secrets.token_urlsafe(16))   # iguala el tiempo de respuesta si el usuario no existe


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verificar_password(hash_guardado: str, password: str) -> bool:
    try:
        return _ph.verify(hash_guardado, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def validar_politica(password: str, email: str = "") -> None:
    faltas = []
    if len(password) < 10:
        faltas.append("al menos 10 caracteres")
    if len(password) > 128:
        faltas.append("como máximo 128 caracteres")
    if not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
        faltas.append("letras y números")
    usuario = email.split("@")[0].lower()
    if usuario and len(usuario) >= 3 and usuario in password.lower():
        faltas.append("no contener tu nombre de usuario")
    if faltas:
        raise HTTPException(422, "La contraseña debe tener " + ", ".join(faltas) + ".")


def password_temporal(n: int = 14) -> str:
    alfabeto = string.ascii_letters + string.digits
    while True:
        p = "".join(secrets.choice(alfabeto) for _ in range(n))
        if any(c.isalpha() for c in p) and any(c.isdigit() for c in p):
            return p


# --------------------------------------------------------------------------- utilidades
def ip_cliente(request: Request | None) -> str | None:
    if request is None:
        return None
    xff = request.headers.get("x-forwarded-for")       # lo agrega el BFF (xfwd: true)
    if xff:
        return xff.split(",")[0].strip()[:64]
    return request.client.host if request.client else None


def registrar(dbs, accion: str, request: Request | None = None, usuario: Usuario | None = None,
              email: str | None = None, **detalle) -> None:
    dbs.add(Evento(usuario_id=usuario.id if usuario else None,
                   email=email or (usuario.email if usuario else None),
                   accion=accion, ip=ip_cliente(request),
                   detalle=json.dumps(detalle, ensure_ascii=False) if detalle else None))
    dbs.commit()


def iso_utc(fecha) -> str | None:
    """Fecha guardada (UTC sin zona) -> ISO con "Z", para que el navegador la convierta bien."""
    return fecha.isoformat(timespec="seconds") + "Z" if fecha else None


def usuario_publico(u: Usuario) -> dict:
    return {"id": u.id, "email": u.email, "nombre": u.nombre, "rol": u.rol, "activo": u.activo,
            "debe_cambiar_password": u.debe_cambiar_password,
            "bloqueado": bool(u.bloqueado_hasta and u.bloqueado_hasta > ahora()),
            "creado_en": iso_utc(u.creado_en), "ultimo_acceso": iso_utc(u.ultimo_acceso)}


_intentos_ip: dict[str, deque] = defaultdict(deque)


def limitar_intentos_ip(ip: str | None) -> None:
    clave, t = ip or "?", time.monotonic()
    cola = _intentos_ip[clave]
    while cola and t - cola[0] > VENTANA_IP_S:
        cola.popleft()
    if len(cola) >= _cfg()["login_por_ip"]:
        raise HTTPException(429, "Demasiados intentos de inicio de sesión. Espera unos minutos.")
    cola.append(t)


def verificar_csrf(request: Request) -> None:
    if request.method not in ("GET", "HEAD", "OPTIONS") and \
            request.headers.get(CSRF_HEADER) != CSRF_VALOR:
        raise HTTPException(403, "Petición rechazada: falta el encabezado anti-CSRF.")


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# --------------------------------------------------------------------------- sesiones
def crear_sesion(dbs, usuario: Usuario, request: Request, response: Response) -> None:
    cfg, token = _cfg(), secrets.token_urlsafe(32)
    t = ahora()
    dbs.add(Sesion(usuario_id=usuario.id, token_hash=_hash_token(token), creada_en=t,
                   expira_en=t + cfg["duracion_max"], ultimo_uso=t, ip=ip_cliente(request),
                   user_agent=(request.headers.get("user-agent") or "")[:200]))
    dbs.commit()
    response.set_cookie(COOKIE, token, httponly=True, secure=cfg["cookie_secure"], samesite="strict",
                        max_age=int(cfg["duracion_max"].total_seconds()), path="/")


def revocar_sesiones(dbs, usuario_id: int, excepto: int | None = None) -> None:
    q = update(Sesion).where(Sesion.usuario_id == usuario_id, Sesion.revocada.is_(False))
    if excepto is not None:
        q = q.where(Sesion.id != excepto)
    dbs.execute(q.values(revocada=True))
    dbs.commit()


def revocar_sesion_actual(dbs, request: Request) -> Sesion | None:
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    s = dbs.scalar(select(Sesion).where(Sesion.token_hash == _hash_token(token)))
    if s and not s.revocada:
        s.revocada = True
        dbs.commit()
    return s


def _usuario_de_la_sesion(request: Request, dbs, permitir_cambio_pendiente: bool) -> Usuario:
    verificar_csrf(request)
    token = request.cookies.get(COOKIE)
    if not token:
        raise HTTPException(401, "No has iniciado sesión.")
    s = dbs.scalar(select(Sesion).where(Sesion.token_hash == _hash_token(token)))
    t, cfg = ahora(), _cfg()
    if s is None or s.revocada or s.expira_en <= t or t - s.ultimo_uso > cfg["inactividad"]:
        if s is not None and not s.revocada:
            s.revocada = True
            dbs.commit()
        raise HTTPException(401, "Tu sesión expiró. Inicia sesión de nuevo.")
    u = dbs.get(Usuario, s.usuario_id)
    if u is None or not u.activo:
        raise HTTPException(401, "Tu sesión ya no es válida.")
    if u.debe_cambiar_password and not permitir_cambio_pendiente:
        raise HTTPException(403, {"codigo": "cambio_password_requerido",
                                  "mensaje": "Debes cambiar tu contraseña antes de continuar."})
    if t - s.ultimo_uso > timedelta(minutes=1):           # no escribir en cada request
        s.ultimo_uso = t
        dbs.commit()
    request.state.sesion_id = s.id
    return u


# --------------------------------------------------------------------------- dependencias
def obtener_db():
    yield from db.obtener_sesion()


def usuario_actual(request: Request, dbs=Depends(obtener_db)) -> Usuario:
    return _usuario_de_la_sesion(request, dbs, permitir_cambio_pendiente=False)


def usuario_con_cambio_pendiente(request: Request, dbs=Depends(obtener_db)) -> Usuario:
    """Para /auth/me, /auth/cambiar-password: accesibles aunque falte cambiar la contraseña."""
    return _usuario_de_la_sesion(request, dbs, permitir_cambio_pendiente=True)


def requiere_rol(*roles: str):
    def dependencia(u: Usuario = Depends(usuario_actual)) -> Usuario:
        if u.rol not in roles:
            raise HTTPException(403, "No tienes permiso para esta acción.")
        return u
    return dependencia


solo_admin = requiere_rol("admin")
puede_evaluar = requiere_rol("admin", "clinico")


def acceso_evaluacion(accion: str):
    """Exige rol para evaluar y deja constancia en la auditoría (sin datos clínicos)."""
    def dependencia(request: Request, u: Usuario = Depends(puede_evaluar), dbs=Depends(obtener_db)) -> Usuario:
        extra = {"eje": request.query_params["eje"]} if "eje" in request.query_params else {}
        registrar(dbs, accion, request, usuario=u, **extra)
        return u
    return dependencia


# --------------------------------------------------------------------------- admin inicial
def crear_admin_inicial(dbs) -> Usuario | None:
    """Si no existe ningún administrador, crea uno con ADMIN_EMAIL / ADMIN_PASSWORD. Sin
    ADMIN_PASSWORD genera una contraseña aleatoria y la muestra UNA vez en el log. En ambos
    casos exige cambiarla en el primer ingreso."""
    if dbs.scalar(select(func.count()).select_from(Usuario).where(Usuario.rol == "admin")):
        return None
    email = os.environ.get("ADMIN_EMAIL", "admin@plataforma.local").strip().lower()
    password = os.environ.get("ADMIN_PASSWORD") or password_temporal(16)
    u = Usuario(email=email, nombre="Administrador", rol="admin", hash_password=hash_password(password),
                activo=True, debe_cambiar_password=True)
    dbs.add(u)
    dbs.commit()
    registrar(dbs, "admin_inicial_creado", usuario=u)
    linea = "=" * 70
    print(f"\n{linea}\n[seguridad] Administrador inicial creado: {email}")
    if os.environ.get("ADMIN_PASSWORD"):
        print("[seguridad] Contraseña: la definida en ADMIN_PASSWORD.")
    else:
        print(f"[seguridad] Contraseña temporal: {password}")
    print(f"[seguridad] Se pedirá cambiarla en el primer ingreso.\n{linea}\n", flush=True)
    return u
