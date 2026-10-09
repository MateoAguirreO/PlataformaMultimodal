"""Pruebas de autenticación, sesiones, roles y auditoría.

Corren contra una app mínima (rutas /auth y /admin + un endpoint protegido) con una base
SQLite temporal, sin cargar los modelos. La última prueba usa la app real (requiere las
dependencias pesadas del backend; se omite si faltan).

Uso:  cd backend && python -m pytest -q
"""
import sys
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import db  # noqa: E402
import seguridad as S  # noqa: E402
from db import Usuario  # noqa: E402
from rutas_auth import router_admin, router_auth  # noqa: E402

H = {"X-Requested-With": "fetch"}
ADMIN, CLAVE_INICIAL, CLAVE_NUEVA = "admin@test.local", "Inicial12345", "NuevaClave2026"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_EMAIL", ADMIN)
    monkeypatch.setenv("ADMIN_PASSWORD", CLAVE_INICIAL)
    monkeypatch.delenv("SESION_INACTIVIDAD_MIN", raising=False)
    db.configurar(f"sqlite:///{tmp_path / 'prueba.db'}")
    with db.SessionLocal() as s:
        S.crear_admin_inicial(s)
    S._intentos_ip.clear()
    a = FastAPI()
    a.include_router(router_auth)
    a.include_router(router_admin)

    @a.post("/protegido", dependencies=[Depends(S.acceso_evaluacion("prueba"))])
    def protegido():
        return {"ok": True}

    return a


def login(c, email, password):
    return c.post("/auth/login", json={"email": email, "password": password}, headers=H)


def admin_listo(app):
    c = TestClient(app)
    assert login(c, ADMIN, CLAVE_INICIAL).status_code == 200
    r = c.post("/auth/cambiar-password", json={"actual": CLAVE_INICIAL, "nueva": CLAVE_NUEVA}, headers=H)
    assert r.status_code == 200, r.text
    return c


def crear_clinico(c_admin, email="clinico@test.local"):
    r = c_admin.post("/admin/usuarios", json={"email": email, "nombre": "Clínico", "rol": "clinico"}, headers=H)
    assert r.status_code == 201, r.text
    return r.json()


def test_admin_inicial_debe_cambiar_password(app):
    c = TestClient(app)
    r = login(c, ADMIN, CLAVE_INICIAL)
    assert r.status_code == 200 and r.json()["debe_cambiar_password"] is True
    assert c.get("/auth/me").status_code == 200
    r = c.post("/protegido", headers=H)
    assert r.status_code == 403 and r.json()["detail"]["codigo"] == "cambio_password_requerido"


def test_cambio_password_habilita_acceso_y_audita(app):
    c = admin_listo(app)
    assert c.post("/protegido", headers=H).status_code == 200
    acciones = [e["accion"] for e in c.get("/admin/auditoria").json()]
    assert {"admin_inicial_creado", "login_ok", "cambio_password", "prueba"} <= set(acciones)


def test_politica_de_password(app):
    c = TestClient(app)
    login(c, ADMIN, CLAVE_INICIAL)
    for debil in ("corta1", "sololetrasmuylarga", "12345678901", "admin12345678"):
        r = c.post("/auth/cambiar-password", json={"actual": CLAVE_INICIAL, "nueva": debil}, headers=H)
        assert r.status_code == 422, (debil, r.text)


def test_csrf_obligatorio(app):
    c = TestClient(app)
    assert c.post("/auth/login", json={"email": ADMIN, "password": CLAVE_INICIAL}).status_code == 403
    c = admin_listo(app)
    assert c.post("/protegido").status_code == 403          # sin encabezado
    assert c.post("/protegido", headers=H).status_code == 200


def test_sin_sesion_401(app):
    c = TestClient(app)
    assert c.get("/auth/me").status_code == 401
    assert c.post("/protegido", headers=H).status_code == 401
    assert c.get("/admin/usuarios").status_code == 401


def test_bloqueo_por_intentos_y_mensaje_generico(app):
    c = TestClient(app)
    r_inexistente = login(c, "nadie@test.local", "loquesea123")
    for _ in range(S.MAX_INTENTOS):
        r = login(c, ADMIN, "incorrecta123")
        assert r.status_code == 401
    assert r.json()["detail"] == r_inexistente.json()["detail"] == S.MENSAJE_LOGIN
    assert login(c, ADMIN, CLAVE_INICIAL).status_code == 401       # bloqueada aunque la clave sea correcta
    with db.SessionLocal() as s:
        u = s.query(Usuario).filter_by(email=ADMIN).one()
        assert u.bloqueado_hasta is not None


def test_roles_clinico_no_administra(app):
    ca = admin_listo(app)
    nuevo = crear_clinico(ca)
    assert nuevo["debe_cambiar_password"] and len(nuevo["password_temporal"]) >= 12
    cc = TestClient(app)
    assert login(cc, "clinico@test.local", nuevo["password_temporal"]).status_code == 200
    assert cc.post("/protegido", headers=H).status_code == 403    # primero debe cambiar la clave
    assert cc.post("/auth/cambiar-password", json={"actual": nuevo["password_temporal"],
                                                   "nueva": "Consulta2026Samana"}, headers=H).status_code == 200
    assert cc.post("/protegido", headers=H).status_code == 200
    assert cc.get("/admin/usuarios").status_code == 403
    assert cc.post("/admin/usuarios", json={"email": "x@test.local", "nombre": "X", "rol": "admin"},
                   headers=H).status_code == 403


def test_reset_password_revoca_sesiones(app):
    ca = admin_listo(app)
    nuevo = crear_clinico(ca)
    cc = TestClient(app)
    login(cc, "clinico@test.local", nuevo["password_temporal"])
    assert cc.get("/auth/me").status_code == 200
    r = ca.post(f"/admin/usuarios/{nuevo['id']}/reset-password", headers=H)
    assert r.status_code == 200 and r.json()["password_temporal"] != nuevo["password_temporal"]
    assert cc.get("/auth/me").status_code == 401                      # sesión revocada


def test_desactivar_y_ultimo_admin(app):
    ca = admin_listo(app)
    yo = ca.get("/auth/me").json()
    assert ca.patch(f"/admin/usuarios/{yo['id']}", json={"activo": False}, headers=H).status_code == 409
    assert ca.patch(f"/admin/usuarios/{yo['id']}", json={"rol": "clinico"}, headers=H).status_code == 409
    nuevo = crear_clinico(ca)
    cc = TestClient(app)
    login(cc, "clinico@test.local", nuevo["password_temporal"])
    assert ca.patch(f"/admin/usuarios/{nuevo['id']}", json={"activo": False}, headers=H).status_code == 200
    assert cc.get("/auth/me").status_code == 401
    assert login(TestClient(app), "clinico@test.local", nuevo["password_temporal"]).status_code == 401


def test_logout_revoca_el_token(app):
    c = admin_listo(app)
    token = c.cookies.get(S.COOKIE)
    assert c.post("/auth/logout", headers=H).status_code == 200
    otro = TestClient(app)
    otro.cookies.set(S.COOKIE, token)
    assert otro.get("/auth/me").status_code == 401                    # el token robado ya no sirve


def test_login_nuevo_cierra_la_sesion_previa(app):
    c = admin_listo(app)
    anterior = c.cookies.get(S.COOKIE)
    assert login(c, ADMIN, CLAVE_NUEVA).status_code == 200
    assert c.cookies.get(S.COOKIE) != anterior
    viejo = TestClient(app)
    viejo.cookies.set(S.COOKIE, anterior)
    assert viejo.get("/auth/me").status_code == 401
    assert c.get("/auth/me").status_code == 200


def test_expira_por_inactividad(app, monkeypatch):
    c = admin_listo(app)
    monkeypatch.setenv("SESION_INACTIVIDAD_MIN", "0")
    assert c.get("/auth/me").status_code == 401


def test_cookie_segura(app):
    c = TestClient(app)
    r = login(c, ADMIN, CLAVE_INICIAL)
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    with db.SessionLocal() as s:                                      # en la base solo queda el hash
        assert c.cookies.get(S.COOKIE) not in {x.token_hash for x in s.query(db.Sesion).all()}


def test_fechas_con_zona_utc(app):
    c = admin_listo(app)
    yo = c.get("/auth/me").json()
    assert yo["creado_en"].endswith("Z") and yo["ultimo_acceso"].endswith("Z")
    assert all(e["fecha"].endswith("Z") for e in c.get("/admin/auditoria").json())


def test_consola_recupera_admin_bloqueado(app):
    import gestionar_usuarios as G
    c = TestClient(app)
    for _ in range(S.MAX_INTENTOS):
        login(c, ADMIN, "incorrecta123")
    with db.SessionLocal() as s:
        temporal = G.restablecer(s, ADMIN.upper())
        nuevo = G.crear_admin(s, "segunda@test.local", "Segunda Admin")
    r = login(c, ADMIN, temporal)
    assert r.status_code == 200 and r.json()["debe_cambiar_password"] is True
    assert login(TestClient(app), "segunda@test.local", nuevo).json()["rol"] == "admin"
    with pytest.raises(SystemExit):
        with db.SessionLocal() as s:
            G.crear_admin(s, "segunda@test.local", "Duplicada")


def test_integracion_app_real(tmp_path, monkeypatch):
    pytest.importorskip("shap")
    pytest.importorskip("xgboost")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'real.db'}")
    monkeypatch.setenv("ADMIN_EMAIL", ADMIN)
    monkeypatch.setenv("ADMIN_PASSWORD", CLAVE_INICIAL)
    monkeypatch.setenv("PRECARGAR_VOZ", "false")       # no cargar wav2vec2 en las pruebas
    import app as backend
    S._intentos_ip.clear()
    with TestClient(backend.app) as c:
        assert c.get("/health").status_code == 200
        muestra = {"audio_segments": [{"x": 1.0}], "video_features": {"y": 1.0}}
        assert c.post("/predict", json=muestra, headers=H).status_code == 401
        assert c.get("/schema").status_code == 401
        login(c, ADMIN, CLAVE_INICIAL)
        assert c.get("/schema").status_code == 403                    # falta cambiar la clave
        c.post("/auth/cambiar-password", json={"actual": CLAVE_INICIAL, "nueva": CLAVE_NUEVA}, headers=H)
        assert c.get("/schema").status_code == 200
        assert c.post("/predict", json=muestra, headers=H).status_code == 422   # autenticado; muestra inválida
