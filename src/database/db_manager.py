"""
DatabaseManager
===============
Persistence layer for the Facial Recognition System.

Supports SQLite (default) and PostgreSQL via SQLAlchemy.
Set the ``DATABASE_URL`` environment variable to switch backends:

    export DATABASE_URL=postgresql+psycopg2://user:password@host/dbname

If not set, a local ``facial_recognition.db`` SQLite file is used.
"""

from __future__ import annotations

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
    """Registered person with face embedding."""

    __tablename__ = "persons"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    name: str = Column(String(255), nullable=False)
    embedding_json: str = Column(Text, nullable=False)  # JSON-encoded list[float]
    registered_at: datetime = Column(DateTime, default=datetime.utcnow)


class DetectionLog(Base):
    """Log of every face detection / recognition event."""

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
    """Thin wrapper around SQLAlchemy that exposes domain-level operations.

    Parameters
    ----------
    db_url:
        SQLAlchemy connection string.  Defaults to the ``DATABASE_URL``
        environment variable or a local SQLite file.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        url = db_url or os.getenv("DATABASE_URL", _DEFAULT_DB_URL)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self._engine = create_engine(url, connect_args=connect_args)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("Database engine created: %s", url)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create all tables if they do not already exist."""
        Base.metadata.create_all(self._engine)
        logger.info("Database schema initialised.")

    # ------------------------------------------------------------------
    # Person CRUD
    # ------------------------------------------------------------------

    def save_person(self, name: str, embedding: List[float]) -> int:
        """Persist a new person record and return its ``id``.

        Parameters
        ----------
        name:
            Display name.
        embedding:
            Face embedding vector (list of floats).

        Returns
        -------
        int
            Auto-generated primary key.
        """
        with self._Session() as session:
            person = Person(
                name=name,
                embedding_json=json.dumps(embedding),
            )
            session.add(person)
            session.commit()
            session.refresh(person)
            logger.info("Saved person '%s' with id=%d.", name, person.id)
            return person.id

    def get_person(self, person_id: int) -> Optional[Dict]:
        """Retrieve a single person by primary key.

        Returns
        -------
        dict or None
            ``{"id", "name", "embedding", "registered_at"}`` or ``None``.
        """
        with self._Session() as session:
            record = session.get(Person, person_id)
            if record is None:
                return None
            return self._person_to_dict(record)

    def get_all_people(self) -> List[Dict]:
        """Return all registered people as a list of dicts."""
        with self._Session() as session:
            records = session.query(Person).all()
            return [self._person_to_dict(r) for r in records]

    def delete_person(self, person_id: int) -> bool:
        """Delete a person record by id.

        Returns
        -------
        bool
            ``True`` if deleted, ``False`` if not found.
        """
        with self._Session() as session:
            record = session.get(Person, person_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            logger.info("Deleted person id=%d.", person_id)
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
        """Insert a detection event log record.

        Returns
        -------
        int
            Auto-generated primary key of the new log entry.
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
        """Return recent detection log entries.

        Parameters
        ----------
        limit:
            Maximum number of records to return.
        offset:
            Number of records to skip (for pagination).
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
        """Return aggregated counts per dominant emotion.

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _person_to_dict(record: Person) -> Dict:
        return {
            "id": record.id,
            "name": record.name,
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
