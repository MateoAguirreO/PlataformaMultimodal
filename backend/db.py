"""Persistencia de usuarios, sesiones y auditoría.

SQLite por defecto (archivo en DATOS_DIR, montado como volumen en docker-compose). Para
otro motor (p. ej. PostgreSQL en producción) basta con definir DATABASE_URL.
Fechas en UTC sin zona horaria (naive), de forma consistente en toda la app.
"""
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def ahora() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Usuario(Base):
    __tablename__ = "usuarios"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    nombre: Mapped[str] = mapped_column(String(120))
    rol: Mapped[str] = mapped_column(String(20))                       # admin | clinico
    hash_password: Mapped[str] = mapped_column(String(255))            # argon2id
    activo: Mapped[bool] = mapped_column(Boolean, default=True)
    debe_cambiar_password: Mapped[bool] = mapped_column(Boolean, default=True)
    intentos_fallidos: Mapped[int] = mapped_column(Integer, default=0)
    bloqueado_hasta: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    creado_en: Mapped[datetime] = mapped_column(DateTime, default=ahora)
    ultimo_acceso: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Sesion(Base):
    """Sesión del lado del servidor: el navegador solo guarda un token aleatorio en una
    cookie httpOnly; aquí se guarda su hash (nunca el token), lo que permite revocarla."""
    __tablename__ = "sesiones"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    usuario_id: Mapped[int] = mapped_column(ForeignKey("usuarios.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    creada_en: Mapped[datetime] = mapped_column(DateTime, default=ahora)
    expira_en: Mapped[datetime] = mapped_column(DateTime)
    ultimo_uso: Mapped[datetime] = mapped_column(DateTime, default=ahora)
    revocada: Mapped[bool] = mapped_column(Boolean, default=False)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(200), nullable=True)


class Evento(Base):
    """Auditoría: quién hizo qué y cuándo. Nunca guarda datos clínicos (features,
    probabilidades) ni contraseñas."""
    __tablename__ = "auditoria"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fecha: Mapped[datetime] = mapped_column(DateTime, default=ahora, index=True)
    usuario_id: Mapped[int | None] = mapped_column(ForeignKey("usuarios.id"), nullable=True, index=True)
    email: Mapped[str | None] = mapped_column(String(254), nullable=True)
    accion: Mapped[str] = mapped_column(String(40), index=True)
    detalle: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)


_engine = None
SessionLocal = sessionmaker(expire_on_commit=False)


def _url_por_defecto() -> str:
    datos = Path(os.environ.get("DATOS_DIR", Path(__file__).resolve().parent / "datos"))
    datos.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{datos / 'plataforma.db'}"


def configurar(url: str | None = None):
    """Crea el engine y las tablas. Se llama al arrancar la app (y en las pruebas, con
    una base temporal)."""
    global _engine
    url = url or os.environ.get("DATABASE_URL") or _url_por_defecto()
    es_sqlite = url.startswith("sqlite")
    _engine = create_engine(url, connect_args={"check_same_thread": False} if es_sqlite else {})
    if es_sqlite:
        @event.listens_for(_engine, "connect")
        def _pragmas(conn, _):
            cur = conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
    SessionLocal.configure(bind=_engine)
    Base.metadata.create_all(_engine)
    return _engine


def obtener_sesion():
    """Dependencia de FastAPI: una sesión de base de datos por request."""
    if _engine is None:
        configurar()
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
