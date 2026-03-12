"""
DatabaseManager
===============
Capa de persistencia del Sistema de Reconocimiento Facial.

Soporta SQLite (por defecto) y PostgreSQL mediante SQLAlchemy.
Configura la variable de entorno ``DATABASE_URL`` para cambiar de backend:

    export DATABASE_URL=postgresql+psycopg2://user:password@host/dbname

Si no se define, se usa un archivo SQLite local ``facial_recognition.db``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLAlchemy ORM models
# ---------------------------------------------------------------------------

_DEFAULT_DB_URL = "sqlite:///facial_recognition.db"


class Base(DeclarativeBase):
    pass


class Person(Base):
    """Persona registrada con embedding facial y datos personales."""

    __tablename__ = "persons"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    name: str = Column(String(255), nullable=False)
    apellido: str = Column(String(255), nullable=False, default="")
    email: str = Column(String(255), nullable=True, unique=True)
    embedding_json: str = Column(Text, nullable=False)  # JSON-encoded list[float]
    registered_at: datetime = Column(DateTime, default=datetime.utcnow)


class DetectionLog(Base):
    """Registro de cada evento de detección / reconocimiento facial."""

    __tablename__ = "detection_logs"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    person_id: Optional[int] = Column(Integer, nullable=True)
    person_name: str = Column(String(255), default="Desconocido")
    dominant_emotion: Optional[str] = Column(String(64), nullable=True)
    confidence: Optional[float] = Column(Float, nullable=True)
    detected_at: datetime = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class DatabaseManager:
    """Wrapper sobre SQLAlchemy que expone operaciones de dominio.

    Parameters
    ----------
    db_url:
        Cadena de conexión de SQLAlchemy. Por defecto usa la variable
        de entorno ``DATABASE_URL`` o un archivo SQLite local.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        url = db_url or os.getenv("DATABASE_URL", _DEFAULT_DB_URL)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self._engine = create_engine(url, connect_args=connect_args)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("Motor de base de datos creado: %s", url)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Crear todas las tablas si no existen."""
        Base.metadata.create_all(self._engine)
        logger.info("Esquema de base de datos inicializado.")

    # ------------------------------------------------------------------
    # Person CRUD
    # ------------------------------------------------------------------

    def save_person(
        self,
        name: str,
        embedding: List[float],
        apellido: str = "",
        email: Optional[str] = None,
    ) -> int:
        """Persistir una nueva persona y devolver su ``id``.

        Parameters
        ----------
        name:
            Nombre de la persona.
        embedding:
            Vector de embedding facial (lista de floats).
        apellido:
            Apellido de la persona.
        email:
            Correo electrónico (opcional, único).

        Returns
        -------
        int
            Clave primaria auto-generada.
        """
        with self._Session() as session:
            person = Person(
                name=name,
                apellido=apellido,
                email=email if email else None,
                embedding_json=json.dumps(embedding),
            )
            session.add(person)
            session.commit()
            session.refresh(person)
            logger.info("Persona '%s %s' guardada con id=%d.", name, apellido, person.id)
            return person.id

    def person_exists(self, email: str) -> bool:
        """Verificar si ya existe una persona con el email dado.

        Parameters
        ----------
        email:
            Correo electrónico a verificar.

        Returns
        -------
        bool
            ``True`` si ya existe un registro con ese email.
        """
        if not email:
            return False
        with self._Session() as session:
            record = session.query(Person).filter(Person.email == email).first()
            return record is not None

    def get_person(self, person_id: int) -> Optional[Dict]:
        """Recuperar una persona por su clave primaria.

        Returns
        -------
        dict or None
            ``{"id", "name", "apellido", "email", "embedding", "registered_at"}``
            o ``None``.
        """
        with self._Session() as session:
            record = session.get(Person, person_id)
            if record is None:
                return None
            return self._person_to_dict(record)

    def get_all_people(self) -> List[Dict]:
        """Devolver todas las personas registradas como lista de dicts."""
        with self._Session() as session:
            records = session.query(Person).all()
            return [self._person_to_dict(r) for r in records]

    def delete_person(self, person_id: int) -> bool:
        """Eliminar una persona por su id.

        Returns
        -------
        bool
            ``True`` si fue eliminada, ``False`` si no se encontró.
        """
        with self._Session() as session:
            record = session.get(Person, person_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            logger.info("Persona id=%d eliminada.", person_id)
            return True

    # ------------------------------------------------------------------
    # Detection log CRUD
    # ------------------------------------------------------------------

    def log_detection(
        self,
        person_id: Optional[int] = None,
        person_name: str = "Desconocido",
        dominant_emotion: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """Insertar un registro de evento de detección.

        Returns
        -------
        int
            Clave primaria del nuevo registro.
        """
        with self._Session() as session:
            log = DetectionLog(
                person_id=person_id,
                person_name=person_name,
                dominant_emotion=dominant_emotion,
                confidence=confidence,
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            return log.id

    def get_detection_logs(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """Devolver registros recientes de detección.

        Parameters
        ----------
        limit:
            Número máximo de registros.
        offset:
            Registros a saltar (para paginación).
        """
        with self._Session() as session:
            records = (
                session.query(DetectionLog)
                .order_by(DetectionLog.detected_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [self._log_to_dict(r) for r in records]

    def get_emotion_summary(self) -> List[Dict]:
        """Devolver conteos agregados por emoción dominante.

        Returns
        -------
        list of dict
            ``[{"emotion": str, "count": int}, ...]``
        """
        with self._Session() as session:
            rows = session.execute(
                text(
                    "SELECT dominant_emotion, COUNT(*) AS cnt "
                    "FROM detection_logs "
                    "WHERE dominant_emotion IS NOT NULL "
                    "GROUP BY dominant_emotion "
                    "ORDER BY cnt DESC"
                )
            ).fetchall()
            return [{"emotion": row[0], "count": row[1]} for row in rows]

    def get_emotions_by_person(self, person_id: int) -> List[Dict]:
        """Devolver distribución de emociones para una persona específica.

        Parameters
        ----------
        person_id:
            ID de la persona registrada.

        Returns
        -------
        list of dict
            ``[{"emotion": str, "count": int}, ...]``
        """
        with self._Session() as session:
            rows = session.execute(
                text(
                    "SELECT dominant_emotion, COUNT(*) AS cnt "
                    "FROM detection_logs "
                    "WHERE person_id = :pid AND dominant_emotion IS NOT NULL "
                    "GROUP BY dominant_emotion "
                    "ORDER BY cnt DESC"
                ),
                {"pid": person_id},
            ).fetchall()
            return [{"emotion": row[0], "count": row[1]} for row in rows]

    def get_person_detection_stats(self) -> List[Dict]:
        """Devolver estadísticas de detección por persona.

        Returns
        -------
        list of dict
            ``[{"person_name": str, "count": int, "person_id": int|None}, ...]``
        """
        with self._Session() as session:
            rows = session.execute(
                text(
                    "SELECT person_name, person_id, COUNT(*) AS cnt "
                    "FROM detection_logs "
                    "GROUP BY person_name, person_id "
                    "ORDER BY cnt DESC"
                )
            ).fetchall()
            return [
                {"person_name": row[0], "person_id": row[1], "count": row[2]}
                for row in rows
            ]

    def get_general_stats(self) -> Dict:
        """Devolver estadísticas generales del sistema.

        Returns
        -------
        dict
            ``{"total_detections", "total_persons", "most_frequent_person",
               "most_frequent_emotion"}``
        """
        with self._Session() as session:
            # Total de detecciones
            total_det = session.execute(
                text("SELECT COUNT(*) FROM detection_logs")
            ).scalar() or 0

            # Total de personas registradas
            total_persons = session.execute(
                text("SELECT COUNT(*) FROM persons")
            ).scalar() or 0

            # Persona más frecuente
            top_person_row = session.execute(
                text(
                    "SELECT person_name, COUNT(*) AS cnt "
                    "FROM detection_logs "
                    "WHERE person_name != 'Desconocido' "
                    "GROUP BY person_name "
                    "ORDER BY cnt DESC LIMIT 1"
                )
            ).first()

            # Emoción más frecuente
            top_emotion_row = session.execute(
                text(
                    "SELECT dominant_emotion, COUNT(*) AS cnt "
                    "FROM detection_logs "
                    "WHERE dominant_emotion IS NOT NULL "
                    "GROUP BY dominant_emotion "
                    "ORDER BY cnt DESC LIMIT 1"
                )
            ).first()

            return {
                "total_detections": total_det,
                "total_persons": total_persons,
                "most_frequent_person": top_person_row[0] if top_person_row else "—",
                "most_frequent_emotion": top_emotion_row[0] if top_emotion_row else "—",
            }

    def export_logs_csv(self, filepath: str) -> int:
        """Exportar todo el historial de detecciones a un archivo CSV.

        Parameters
        ----------
        filepath:
            Ruta del archivo CSV de salida.

        Returns
        -------
        int
            Número de registros exportados.
        """
        logs = self.get_detection_logs(limit=100_000, offset=0)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "person_id", "person_name",
                            "dominant_emotion", "confidence", "detected_at"],
            )
            writer.writeheader()
            writer.writerows(logs)
        logger.info("Exportados %d registros a '%s'.", len(logs), filepath)
        return len(logs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _person_to_dict(record: Person) -> Dict:
        return {
            "id": record.id,
            "name": record.name,
            "apellido": record.apellido or "",
            "email": record.email or "",
            "embedding": json.loads(record.embedding_json),
            "registered_at": record.registered_at,
        }

    @staticmethod
    def _log_to_dict(record: DetectionLog) -> Dict:
        return {
            "id": record.id,
            "person_id": record.person_id,
            "person_name": record.person_name,
            "dominant_emotion": record.dominant_emotion,
            "confidence": record.confidence,
            "detected_at": record.detected_at,
        }
