"""
EmotionAnalyzer
===============
Classifies the seven basic emotions in a face image using DeepFace.

The seven basic emotions are:
    angry, disgust, fear, happy, sad, surprise, neutral
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTIONS: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# Human-readable labels in Spanish (as requested in the problem statement)
EMOTION_LABELS_ES: Dict[str, str] = {
    "angry": "Enojo",
    "disgust": "Disgusto",
    "fear": "Miedo",
    "happy": "Felicidad",
    "neutral": "Neutral",
    "sad": "Tristeza",
    "surprise": "Sorpresa",
}


class EmotionAnalyzer:
    """Analyse facial expressions and return a probability distribution.

    Parameters
    ----------
    detector_backend:
        Face detector used internally by DeepFace. Supported values:
        ``"opencv"``, ``"ssd"``, ``"mtcnn"``, ``"retinaface"``,
        ``"mediapipe"``.
    """

    def __init__(self, detector_backend: str = "opencv") -> None:
        self.detector_backend = detector_backend

        try:
            from deepface import DeepFace  # type: ignore
            self._deepface = DeepFace
            logger.info("DeepFace emotion backend loaded.")
        except ImportError:
            self._deepface = None
            logger.warning(
                "DeepFace is not installed. Emotion analysis will not work."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, face_img: np.ndarray) -> Optional[Dict[str, object]]:
        """Analyse *face_img* and return emotion scores.

        Parameters
        ----------
        face_img:
            BGR or RGB image of the face region.

        Returns
        -------
        dict or None
            ``{
                "dominant_emotion": str,
                "dominant_emotion_es": str,
                "scores": {emotion: probability},
                "scores_es": {spanish_label: probability},
            }``
            or ``None`` on failure.

        Example
        -------
        ::

            analyzer = EmotionAnalyzer()
            result = analyzer.analyze(face_img)
            print(result["dominant_emotion"])   # "happy"
            print(result["dominant_emotion_es"]) # "Felicidad"
        """
        if self._deepface is None:
            raise RuntimeError("DeepFace is not installed.")

        try:
            results = self._deepface.analyze(
                img_path=face_img,
                actions=["emotion"],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True,
            )
        except Exception as exc:
            logger.error("Emotion analysis failed: %s", exc)
            return None

        if not results:
            return None

        # DeepFace.analyze can return a list or a single dict
        result = results[0] if isinstance(results, list) else results

        raw_scores: Dict[str, float] = result.get("emotion", {})
        dominant: str = result.get("dominant_emotion", "neutral")

        # Normalise scores so they sum to 1.0
        total = sum(raw_scores.values()) or 1.0
        normalised_scores = {k: v / total for k, v in raw_scores.items()}

        scores_es = {
            EMOTION_LABELS_ES.get(k, k): v
            for k, v in normalised_scores.items()
        }

        return {
            "dominant_emotion": dominant,
            "dominant_emotion_es": EMOTION_LABELS_ES.get(dominant, dominant),
            "scores": normalised_scores,
            "scores_es": scores_es,
        }

    def classify(self, face_img: np.ndarray) -> Optional[str]:
        """Return only the dominant emotion label for *face_img*.

        Parameters
        ----------
        face_img:
            BGR or RGB image of the face region.

        Returns
        -------
        str or None
            Dominant emotion in English (e.g. ``"happy"``), or ``None``.
        """
        result = self.analyze(face_img)
        return result["dominant_emotion"] if result else None

    def classify_es(self, face_img: np.ndarray) -> Optional[str]:
        """Return the dominant emotion label in Spanish.

        Parameters
        ----------
        face_img:
            BGR or RGB image of the face region.

        Returns
        -------
        str or None
            Dominant emotion in Spanish (e.g. ``"Felicidad"``), or ``None``.
        """
        result = self.analyze(face_img)
        return result["dominant_emotion_es"] if result else None

    @staticmethod
    def get_emotion_label_es(emotion: str) -> str:
        """Translate an English emotion key to its Spanish label.

        Parameters
        ----------
        emotion:
            English emotion key (e.g. ``"happy"``).

        Returns
        -------
        str
            Spanish label or the original key if not found.
        """
        return EMOTION_LABELS_ES.get(emotion, emotion)

    @staticmethod
    def list_emotions() -> List[str]:
        """Return the list of all recognised emotion keys (English)."""
        return list(EMOTIONS)
