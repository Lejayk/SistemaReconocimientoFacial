"""
FaceRecognizer
==============
Maneja la detección de rostros, extracción de embeddings y
comparación de identidades.

Dependencias: OpenCV, DeepFace (o modelo TensorFlow/ONNX).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alias de tipos
# ---------------------------------------------------------------------------
Embedding = List[float]


class FaceRecognizer:
    """Detectar rostros en frames, construir embeddings y comparar identidades.

    Parameters
    ----------
    model_name:
        Modelo de DeepFace para extracción de embeddings.
        Valores soportados: ``"VGG-Face"``, ``"Facenet"``, ``"Facenet512"``,
        ``"OpenFace"``, ``"DeepFace"``, ``"ArcFace"``.
    detector_backend:
        Detector facial usado internamente por DeepFace.
        Valores soportados: ``"opencv"``, ``"ssd"``, ``"mtcnn"``,
        ``"retinaface"``, ``"mediapipe"``.
    distance_metric:
        Métrica de similitud para comparación.
        Valores soportados: ``"cosine"``, ``"euclidean"``,
        ``"euclidean_l2"``.
    threshold:
        Distancia máxima para considerar el mismo rostro.
        ``None`` usa el valor por defecto de DeepFace para el modelo elegido.
    """

    # Tamaño mínimo de rostro aceptable (píxeles)
    _MIN_FACE_SIZE = 80
    # Umbral mínimo de nitidez (varianza del Laplaciano)
    _MIN_SHARPNESS = 50.0

    def __init__(
        self,
        model_name: str = "Facenet512",
        detector_backend: str = "opencv",
        distance_metric: str = "cosine",
        threshold: Optional[float] = None,
    ) -> None:
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.threshold = threshold

        # Importación lazy de DeepFace para permitir importar el módulo
        # incluso cuando no están instaladas todas las dependencias.
        try:
            from deepface import DeepFace  # type: ignore
            self._deepface = DeepFace
            logger.info("DeepFace cargado (modelo=%s)", self.model_name)
        except ImportError:
            self._deepface = None
            logger.warning(
                "DeepFace no está instalado. El reconocimiento facial no funcionará."
            )

    # ------------------------------------------------------------------
    # API Pública
    # ------------------------------------------------------------------

    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """Detectar todos los rostros en *frame* y devolver sus bounding boxes.

        Parameters
        ----------
        frame:
            Imagen BGR como array de NumPy (como la devuelve ``cv2.VideoCapture``).

        Returns
        -------
        list of dict
            Cada entrada tiene las claves ``"x"``, ``"y"``, ``"w"``, ``"h"``
            (coordenadas en píxeles de la región facial) y ``"confidence"`` (float).
        """
        if self._deepface is None:
            raise RuntimeError("DeepFace no está instalado.")

        try:
            results = self._deepface.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )
        except Exception as exc:
            logger.error("La detección de rostros falló: %s", exc)
            return []

        faces = []
        for result in results:
            region = result.get("facial_area", {})
            faces.append(
                {
                    "x": region.get("x", 0),
                    "y": region.get("y", 0),
                    "w": region.get("w", 0),
                    "h": region.get("h", 0),
                    "confidence": result.get("confidence", 0.0),
                }
            )
        return faces

    def get_embedding(self, face_img: np.ndarray) -> Optional[Embedding]:
        """Generar un vector de embedding para un rostro recortado.

        Parameters
        ----------
        face_img:
            Imagen BGR de la región del rostro (recortada).

        Returns
        -------
        list of float or None
            Vector de embedding 128-D / 512-D, o ``None`` en caso de fallo.
        """
        if self._deepface is None:
            raise RuntimeError("DeepFace no está instalado.")

        try:
            result = self._deepface.represent(
                img_path=face_img,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )
            # DeepFace.represent devuelve una lista; tomar el primer resultado.
            return result[0]["embedding"] if result else None
        except Exception as exc:
            logger.error("La extracción de embedding falló: %s", exc)
            return None

    def assess_face_quality(self, face_img: np.ndarray) -> dict:
        """Evaluar la calidad de una imagen de rostro.

        Verifica tamaño mínimo y nitidez para determinar si la imagen
        es apta para registro.

        Parameters
        ----------
        face_img:
            Imagen BGR del rostro recortado.

        Returns
        -------
        dict
            ``{"is_valid": bool, "size_ok": bool, "sharpness_ok": bool,
               "score": float, "width": int, "height": int,
               "sharpness": float}``
        """
        h, w = face_img.shape[:2]
        size_ok = w >= self._MIN_FACE_SIZE and h >= self._MIN_FACE_SIZE

        # Calcular nitidez usando varianza del Laplaciano
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_ok = sharpness >= self._MIN_SHARPNESS

        # Score compuesto (0 a 100)
        size_score = min(w, h) / self._MIN_FACE_SIZE  # >= 1.0 es bueno
        sharp_score = sharpness / self._MIN_SHARPNESS  # >= 1.0 es bueno
        score = min(100.0, (min(size_score, 2.0) / 2.0 * 50) + (min(sharp_score, 2.0) / 2.0 * 50))

        is_valid = size_ok and sharpness_ok

        return {
            "is_valid": is_valid,
            "size_ok": size_ok,
            "sharpness_ok": sharpness_ok,
            "score": round(score, 1),
            "width": w,
            "height": h,
            "sharpness": round(sharpness, 2),
        }

    def register_face(
        self,
        name: str,
        face_img: np.ndarray,
        db_manager,  # Instancia de DatabaseManager (evitar import circular)
        apellido: str = "",
        email: Optional[str] = None,
    ) -> bool:
        """Extraer embedding de *face_img* y persistirlo en la BD.

        Parameters
        ----------
        name:
            Nombre de la persona.
        face_img:
            Imagen BGR del rostro a registrar.
        db_manager:
            Instancia de :class:`~src.database.db_manager.DatabaseManager`.
        apellido:
            Apellido de la persona.
        email:
            Correo electrónico (opcional).

        Returns
        -------
        bool
            ``True`` si el registro fue exitoso, ``False`` en caso contrario.
        """
        embedding = self.get_embedding(face_img)
        if embedding is None:
            logger.warning("No se pudo extraer embedding para '%s'.", name)
            return False

        db_manager.save_person(
            name=name, embedding=embedding, apellido=apellido, email=email
        )
        logger.info("Rostro registrado para '%s %s'.", name, apellido)
        return True

    def register_multiple_faces(
        self,
        name: str,
        apellido: str,
        email: Optional[str],
        face_imgs: List[np.ndarray],
        db_manager,
    ) -> bool:
        """Registrar persona usando la mejor captura de múltiples imágenes.

        Evalúa la calidad de cada imagen y usa la de mayor puntuación
        para extraer el embedding y registrar.

        Parameters
        ----------
        name:
            Nombre de la persona.
        apellido:
            Apellido de la persona.
        email:
            Correo electrónico (opcional).
        face_imgs:
            Lista de imágenes BGR de rostros capturados.
        db_manager:
            Instancia de DatabaseManager.

        Returns
        -------
        bool
            ``True`` si el registro fue exitoso.
        """
        if not face_imgs:
            logger.warning("No se proporcionaron imágenes para registrar.")
            return False

        # Seleccionar la imagen con mejor calidad
        best_img = None
        best_score = -1.0
        for img in face_imgs:
            quality = self.assess_face_quality(img)
            if quality["score"] > best_score:
                best_score = quality["score"]
                best_img = img

        if best_img is None:
            return False

        return self.register_face(
            name=name,
            face_img=best_img,
            db_manager=db_manager,
            apellido=apellido,
            email=email,
        )

    def compare_embedding(
        self,
        query_embedding: Embedding,
        stored_embedding: Embedding,
    ) -> float:
        """Calcular la distancia entre dos vectores de embedding.

        Parameters
        ----------
        query_embedding:
            Embedding del rostro a identificar.
        stored_embedding:
            Embedding almacenado en la BD de una persona conocida.

        Returns
        -------
        float
            Valor de distancia. Menor es más similar.
        """
        q = np.array(query_embedding, dtype=np.float64)
        s = np.array(stored_embedding, dtype=np.float64)

        if self.distance_metric == "cosine":
            dot = np.dot(q, s)
            norm = np.linalg.norm(q) * np.linalg.norm(s)
            return float(1.0 - dot / (norm + 1e-10))

        if self.distance_metric in ("euclidean", "euclidean_l2"):
            if self.distance_metric == "euclidean_l2":
                q = q / (np.linalg.norm(q) + 1e-10)
                s = s / (np.linalg.norm(s) + 1e-10)
            return float(np.linalg.norm(q - s))

        raise ValueError(f"Métrica de distancia desconocida: {self.distance_metric}")

    def identify(
        self,
        face_img: np.ndarray,
        db_manager,
    ) -> Optional[dict]:
        """Identificar la persona en *face_img* contra embeddings almacenados.

        Parameters
        ----------
        face_img:
            Imagen BGR del rostro a identificar.
        db_manager:
            Instancia de :class:`~src.database.db_manager.DatabaseManager`.

        Returns
        -------
        dict or None
            ``{"name": str, "apellido": str, "distance": float,
              "person_id": int}`` para la mejor coincidencia bajo el
            umbral, o ``None`` si no hay coincidencia.
        """
        query_emb = self.get_embedding(face_img)
        if query_emb is None:
            return None

        people = db_manager.get_all_people()
        if not people:
            return None

        best_match: Optional[dict] = None
        best_distance = float("inf")

        for person in people:
            distance = self.compare_embedding(query_emb, person["embedding"])
            if distance < best_distance:
                best_distance = distance
                best_match = person

        threshold = self.threshold or self._default_threshold()
        if best_distance <= threshold and best_match is not None:
            return {
                "name": best_match["name"],
                "apellido": best_match.get("apellido", ""),
                "distance": best_distance,
                "person_id": best_match["id"],
            }
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_threshold(self) -> float:
        """Devolver un umbral por defecto razonable para el modelo/métrica elegidos."""
        defaults = {
            ("Facenet512", "cosine"): 0.30,
            ("Facenet", "cosine"): 0.40,
            ("VGG-Face", "cosine"): 0.40,
            ("ArcFace", "cosine"): 0.68,
            ("Facenet512", "euclidean_l2"): 0.30,
            ("Facenet", "euclidean_l2"): 0.10,
        }
        return defaults.get((self.model_name, self.distance_metric), 0.40)

    @staticmethod
    def draw_face_boxes(
        frame: np.ndarray,
        faces: List[dict],
        label: str = "",
        color: tuple = (0, 255, 0),
    ) -> np.ndarray:
        """Dibujar bounding boxes y etiqueta opcional en *frame* (in-place).

        Parameters
        ----------
        frame:
            Imagen BGR a anotar.
        faces:
            Lista de dicts de rostros como los devuelve :meth:`detect_faces`.
        label:
            Texto a dibujar encima del primer bounding box.
        color:
            Tupla de color BGR para el rectángulo.

        Returns
        -------
        np.ndarray
            Frame anotado (mismo objeto que *frame*).
        """
        for face in faces:
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if label:
                # Fondo semi-transparente para el texto
                text_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )[0]
                cv2.rectangle(
                    frame,
                    (x, y - text_size[1] - 10),
                    (x + text_size[0] + 4, y),
                    color,
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )
        return frame
