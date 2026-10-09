"""Rutas de autenticación (/auth) y de administración de usuarios (/admin).

No hay borrado de usuarios: se desactivan, para no romper la trazabilidad de la auditoría.
Las contraseñas temporales (usuario nuevo o restablecimiento) se devuelven UNA sola vez
al administrador y obligan a cambiarlas en el primer ingreso.
"""
import json
import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

import seguridad as S
from db import Evento, Usuario, ahora

router_auth = APIRouter(prefix="/auth", tags=["autenticación"])
router_admin = APIRouter(prefix="/admin", tags=["administración"])

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# --------------------------------------------------------------------------- /auth
class LoginIn(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)


@router_auth.post("/login")
def login(datos: LoginIn, request: Request, response: Response, dbs=Depends(S.obtener_db)):
    S.verificar_csrf(request)
    S.limitar_intentos_ip(S.ip_cliente(request))
    email = datos.email.strip().lower()
    u = dbs.scalar(select(Usuario).where(Usuario.email == email))
    t = ahora()
    if u is not None and u.bloqueado_hasta and u.bloqueado_hasta > t:
        S.verificar_password(S._HASH_SEÑUELO, datos.password)   # mismo tiempo que un login normal
        S.registrar(dbs, "login_rechazado_bloqueo", request, usuario=u)
        raise HTTPException(401, S.MENSAJE_LOGIN)
    ok = S.verificar_password(u.hash_password if u else S._HASH_SEÑUELO, datos.password)
    if u is None or not ok or not u.activo:
        if u is not None and not ok:
            u.intentos_fallidos += 1
            if u.intentos_fallidos >= S.MAX_INTENTOS:
                u.bloqueado_hasta, u.intentos_fallidos = t + S.BLOQUEO, 0
                S.registrar(dbs, "cuenta_bloqueada", request, usuario=u)
            dbs.commit()
        S.registrar(dbs, "login_fallido", request, usuario=u, email=email)
        raise HTTPException(401, S.MENSAJE_LOGIN)
    u.intentos_fallidos, u.bloqueado_hasta, u.ultimo_acceso = 0, None, t
    if S._ph.check_needs_rehash(u.hash_password):          # si cambian los parámetros de argon2
        u.hash_password = S.hash_password(datos.password)
    dbs.commit()
    S.revocar_sesion_actual(dbs, request)                  # si el navegador traía otra sesión, se cierra
    S.crear_sesion(dbs, u, request, response)
    S.registrar(dbs, "login_ok", request, usuario=u)
    return S.usuario_publico(u)


@router_auth.post("/logout")
def logout(request: Request, response: Response, dbs=Depends(S.obtener_db)):
    S.verificar_csrf(request)
    s = S.revocar_sesion_actual(dbs, request)
    if s is not None:
        S.registrar(dbs, "logout", request, usuario=dbs.get(Usuario, s.usuario_id))
    response.delete_cookie(S.COOKIE, path="/")
    return {"ok": True}


@router_auth.get("/me")
def me(u: Usuario = Depends(S.usuario_con_cambio_pendiente)):
    return S.usuario_publico(u)


class CambioPasswordIn(BaseModel):
    actual: str = Field(max_length=256)
    nueva: str = Field(max_length=256)


@router_auth.post("/cambiar-password")
def cambiar_password(datos: CambioPasswordIn, request: Request,
                     u: Usuario = Depends(S.usuario_con_cambio_pendiente), dbs=Depends(S.obtener_db)):
    if not S.verificar_password(u.hash_password, datos.actual):
        S.registrar(dbs, "cambio_password_fallido", request, usuario=u)
        raise HTTPException(400, "La contraseña actual no es correcta.")
    if datos.nueva == datos.actual:
        raise HTTPException(422, "La nueva contraseña debe ser distinta de la actual.")
    S.validar_politica(datos.nueva, u.email)
    u.hash_password, u.debe_cambiar_password = S.hash_password(datos.nueva), False
    dbs.commit()
    S.revocar_sesiones(dbs, u.id, excepto=request.state.sesion_id)    # cierra las demás sesiones
    S.registrar(dbs, "cambio_password", request, usuario=u)
    return S.usuario_publico(u)


# --------------------------------------------------------------------------- /admin
@router_admin.get("/usuarios")
def listar_usuarios(_: Usuario = Depends(S.solo_admin), dbs=Depends(S.obtener_db)):
    return [S.usuario_publico(u) for u in dbs.scalars(select(Usuario).order_by(Usuario.id))]


class NuevoUsuarioIn(BaseModel):
    email: str = Field(max_length=254)
    nombre: str = Field(min_length=1, max_length=120)
    rol: str


@router_admin.post("/usuarios", status_code=201)
def crear_usuario(datos: NuevoUsuarioIn, request: Request, admin: Usuario = Depends(S.solo_admin),
                  dbs=Depends(S.obtener_db)):
    email = datos.email.strip().lower()
    if not _EMAIL.match(email):
        raise HTTPException(422, "Correo electrónico no válido.")
    if datos.rol not in S.ROLES:
        raise HTTPException(422, f"Rol no válido. Opciones: {', '.join(S.ROLES)}.")
    if dbs.scalar(select(Usuario).where(Usuario.email == email)):
        raise HTTPException(409, "Ya existe un usuario con ese correo.")
    temporal = S.password_temporal()
    u = Usuario(email=email, nombre=datos.nombre.strip(), rol=datos.rol,
                hash_password=S.hash_password(temporal), activo=True, debe_cambiar_password=True)
    dbs.add(u)
    dbs.commit()
    S.registrar(dbs, "crear_usuario", request, usuario=admin, objetivo=email, rol=datos.rol)
    return {**S.usuario_publico(u), "password_temporal": temporal}


class CambiosUsuarioIn(BaseModel):
    nombre: str | None = Field(default=None, min_length=1, max_length=120)
    rol: str | None = None
    activo: bool | None = None


def _quedan_admins_activos(dbs, excluyendo: int) -> int:
    return dbs.scalar(select(func.count()).select_from(Usuario).where(
        Usuario.rol == "admin", Usuario.activo.is_(True), Usuario.id != excluyendo))


@router_admin.patch("/usuarios/{uid}")
def actualizar_usuario(uid: int, datos: CambiosUsuarioIn, request: Request,
                       admin: Usuario = Depends(S.solo_admin), dbs=Depends(S.obtener_db)):
    u = dbs.get(Usuario, uid)
    if u is None:
        raise HTTPException(404, "Usuario no encontrado.")
    if datos.rol is not None and datos.rol not in S.ROLES:
        raise HTTPException(422, f"Rol no válido. Opciones: {', '.join(S.ROLES)}.")
    pierde_admin = u.rol == "admin" and ((datos.rol is not None and datos.rol != "admin")
                                         or datos.activo is False)
    if pierde_admin and not _quedan_admins_activos(dbs, u.id):
        raise HTTPException(409, "Debe quedar al menos un administrador activo.")
    cambios = {}
    for campo in ("nombre", "rol", "activo"):
        valor = getattr(datos, campo)
        if valor is not None and valor != getattr(u, campo):
            cambios[campo] = valor
            setattr(u, campo, valor.strip() if isinstance(valor, str) else valor)
    dbs.commit()
    if cambios.get("activo") is False:
        S.revocar_sesiones(dbs, u.id)
    if cambios:
        S.registrar(dbs, "actualizar_usuario", request, usuario=admin, objetivo=u.email, cambios=cambios)
    return S.usuario_publico(u)


@router_admin.post("/usuarios/{uid}/reset-password")
def restablecer_password(uid: int, request: Request, admin: Usuario = Depends(S.solo_admin),
                         dbs=Depends(S.obtener_db)):
    S.verificar_csrf(request)
    u = dbs.get(Usuario, uid)
    if u is None:
        raise HTTPException(404, "Usuario no encontrado.")
    temporal = S.password_temporal()
    u.hash_password, u.debe_cambiar_password = S.hash_password(temporal), True
    u.intentos_fallidos, u.bloqueado_hasta = 0, None
    dbs.commit()
    S.revocar_sesiones(dbs, u.id)
    S.registrar(dbs, "restablecer_password", request, usuario=admin, objetivo=u.email)
    return {"email": u.email, "password_temporal": temporal}


@router_admin.get("/auditoria")
def auditoria(limite: int = Query(200, ge=1, le=2000), accion: str | None = None,
              usuario_id: int | None = None, _: Usuario = Depends(S.solo_admin), dbs=Depends(S.obtener_db)):
    q = select(Evento).order_by(Evento.id.desc()).limit(limite)
    if accion:
        q = q.where(Evento.accion == accion)
    if usuario_id is not None:
        q = q.where(Evento.usuario_id == usuario_id)
    return [{"id": e.id, "fecha": S.iso_utc(e.fecha), "usuario_id": e.usuario_id, "email": e.email,
             "accion": e.accion, "detalle": json.loads(e.detalle) if e.detalle else None, "ip": e.ip}
            for e in dbs.scalars(q)]
