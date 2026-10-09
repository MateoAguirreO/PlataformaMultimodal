"""Gestión de usuarios por consola, para recuperar el acceso cuando no se puede por la web
(p. ej. el único administrador olvidó su contraseña o quedó desactivado).

Con Docker:
  docker compose exec backend python gestionar_usuarios.py listar
  docker compose exec backend python gestionar_usuarios.py restablecer admin@plataforma.local
  docker compose exec backend python gestionar_usuarios.py crear-admin persona@dominio.org "Nombre Apellido"

Sin Docker: lo mismo desde la carpeta backend/ (usa la misma base: DATABASE_URL o DATOS_DIR).
Las contraseñas temporales se muestran una sola vez y hay que cambiarlas al ingresar. Cada
acción queda en la auditoría con origen "consola".
"""
import argparse
import sys

from sqlalchemy import select

import db
import seguridad as S
from db import Usuario
from rutas_auth import _EMAIL


def _buscar(dbs, email: str) -> Usuario:
    u = dbs.scalar(select(Usuario).where(Usuario.email == email.strip().lower()))
    if u is None:
        sys.exit(f"No existe ningún usuario con el correo {email}.")
    return u


def listar(dbs) -> None:
    for u in dbs.scalars(select(Usuario).order_by(Usuario.id)):
        p = S.usuario_publico(u)
        estado = ["activo" if u.activo else "INACTIVO"]
        if p["bloqueado"]:
            estado.append("bloqueado")
        if u.debe_cambiar_password:
            estado.append("debe cambiar contraseña")
        print(f"{u.id:>4}  {u.email:<38} {u.rol:<8} {', '.join(estado)}")


def restablecer(dbs, email: str) -> str:
    """Contraseña temporal nueva; además desbloquea, reactiva y cierra sus sesiones."""
    u = _buscar(dbs, email)
    temporal = S.password_temporal()
    u.hash_password, u.debe_cambiar_password = S.hash_password(temporal), True
    u.intentos_fallidos, u.bloqueado_hasta, u.activo = 0, None, True
    dbs.commit()
    S.revocar_sesiones(dbs, u.id)
    S.registrar(dbs, "restablecer_password", objetivo=u.email, origen="consola")
    return temporal


def crear_admin(dbs, email: str, nombre: str) -> str:
    email = email.strip().lower()
    if not _EMAIL.match(email):
        sys.exit("Correo electrónico no válido.")
    if dbs.scalar(select(Usuario).where(Usuario.email == email)):
        sys.exit(f"Ya existe un usuario con el correo {email} (usa 'restablecer').")
    temporal = S.password_temporal()
    dbs.add(Usuario(email=email, nombre=nombre.strip(), rol="admin", hash_password=S.hash_password(temporal),
                    activo=True, debe_cambiar_password=True))
    dbs.commit()
    S.registrar(dbs, "crear_usuario", objetivo=email, rol="admin", origen="consola")
    return temporal


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Gestión de usuarios de la plataforma (recuperación de acceso).")
    sub = p.add_subparsers(dest="comando", required=True)
    sub.add_parser("listar", help="lista los usuarios y su estado")
    r = sub.add_parser("restablecer", help="contraseña temporal nueva (desbloquea y reactiva)")
    r.add_argument("email")
    c = sub.add_parser("crear-admin", help="crea otro administrador")
    c.add_argument("email")
    c.add_argument("nombre")
    a = p.parse_args(argv)

    db.configurar()
    with db.SessionLocal() as dbs:
        if a.comando == "listar":
            listar(dbs)
            return
        temporal = restablecer(dbs, a.email) if a.comando == "restablecer" else crear_admin(dbs, a.email, a.nombre)
    print(f"Contraseña temporal para {a.email.strip().lower()}: {temporal}")
    print("Se muestra una sola vez; se pedirá cambiarla en el primer ingreso.")


if __name__ == "__main__":
    main()
