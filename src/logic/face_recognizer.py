"""
FaceRecognizer
==============
Handles face detection, embedding extraction and identity matching.

Dependencies: OpenCV, DeepFace (or a raw TensorFlow/ONNX model).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Embedding = List[float]


class FaceRecognizer:
    """Detect faces in frames, build embeddings and compare identities.

    Parameters
    ----------
    model_name:
        DeepFace model to use for embedding extraction.
        Supported values: ``"VGG-Face"``, ``"Facenet"``, ``"Facenet512"``,
        ``"OpenFace"``, ``"DeepFace"``, ``"ArcFace"``.
    detector_backend:
        Face detector used internally by DeepFace.
        Supported values: ``"opencv"``, ``"ssd"``, ``"mtcnn"``,
        ``"retinaface"``, ``"mediapipe"``.
    distance_metric:
        Similarity metric for comparison.
        Supported values: ``"cosine"``, ``"euclidean"``,
        ``"euclidean_l2"``.
    threshold:
        Maximum distance to consider two faces as the same person.
        ``None`` uses DeepFace's built-in default for the chosen model.
    """

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

        # Lazy-import DeepFace to allow the module to be imported even when
        # the full dependency stack is not installed (e.g. during unit tests).
        try:
            from deepface import DeepFace  # type: ignore
            self._deepface = DeepFace
            logger.info("DeepFace loaded (model=%s)", self.model_name)
        except ImportError:
            self._deepface = None
            logger.warning(
                "DeepFace is not installed. Face recognition will not work."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """Detect all faces in *frame* and return their bounding boxes.

        Parameters
        ----------
        frame:
            BGR image as a NumPy array (as returned by ``cv2.VideoCapture``).

        Returns
        -------
        list of dict
            Each entry has keys ``"x"``, ``"y"``, ``"w"``, ``"h"`` (pixel
            coordinates of the face region) plus ``"confidence"`` (float).
        """
        if self._deepface is None:
            raise RuntimeError("DeepFace is not installed.")

        try:
            results = self._deepface.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )
        except Exception as exc:
            logger.error("Face detection failed: %s", exc)
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
        """Generate a feature embedding vector for a single cropped face.

        Parameters
        ----------
        face_img:
            BGR image of the face region (cropped).

        Returns
        -------
        list of float or None
            128-D / 512-D embedding vector, or ``None`` on failure.
        """
        if self._deepface is None:
            raise RuntimeError("DeepFace is not installed.")

        try:
            result = self._deepface.represent(
                img_path=face_img,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )
            # DeepFace.represent returns a list; take the first result.
            return result[0]["embedding"] if result else None
        except Exception as exc:
            logger.error("Embedding extraction failed: %s", exc)
            return None

    def register_face(
        self,
        name: str,
        face_img: np.ndarray,
        db_manager,  # DatabaseManager instance (avoid circular import)
    ) -> bool:
        """Extract an embedding from *face_img* and persist it in the DB.

        Parameters
        ----------
        name:
            Person's display name or ID.
        face_img:
            BGR image of the face to register.
        db_manager:
            :class:`~src.database.db_manager.DatabaseManager` instance.

        Returns
        -------
        bool
            ``True`` on success, ``False`` otherwise.
        """
        embedding = self.get_embedding(face_img)
        if embedding is None:
            logger.warning("Could not extract embedding for '%s'.", name)
            return False

        db_manager.save_person(name=name, embedding=embedding)
        logger.info("Registered face for '%s'.", name)
        return True

    def compare_embedding(
        self,
        query_embedding: Embedding,
        stored_embedding: Embedding,
    ) -> float:
        """Compute the distance between two embedding vectors.

        Parameters
        ----------
        query_embedding:
            Embedding of the face to identify.
        stored_embedding:
            Embedding stored in the database for a known person.

        Returns
        -------
        float
            Distance value. Lower is more similar.
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

        raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def identify(
        self,
        face_img: np.ndarray,
        db_manager,
    ) -> Optional[dict]:
        """Identify the person in *face_img* against stored embeddings.

        Parameters
        ----------
        face_img:
            BGR image of the face to identify.
        db_manager:
            :class:`~src.database.db_manager.DatabaseManager` instance.

        Returns
        -------
        dict or None
            ``{"name": str, "distance": float, "person_id": int}`` for the
            best match below the threshold, or ``None`` if no match found.
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
                "distance": best_distance,
                "person_id": best_match["id"],
            }
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_threshold(self) -> float:
        """Return a sensible default threshold for the chosen model/metric."""
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
        """Draw bounding boxes and an optional label on *frame* (in-place).

        Parameters
        ----------
        frame:
            BGR image to annotate.
        faces:
            List of face dicts as returned by :meth:`detect_faces`.
        label:
            Text to draw above the first bounding box.
        color:
            BGR colour tuple for the rectangle.

        Returns
        -------
        np.ndarray
            Annotated frame (same object as *frame*).
        """
        for face in faces:
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if label:
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
        return frame
