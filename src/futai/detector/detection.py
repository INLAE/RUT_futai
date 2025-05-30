"""
Обертка над YOLO для детекции мяча, игроков и судьи
"""

import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.futai.constants import DEFAULT_CONFIDENCE_THRESHOLD


class PlayerDetectionModel:
    """
    Класс-обертка для YOLOv8.
    """

    def __init__(self, weights_path: str, device: str = None):
        self.model = YOLO(weights_path)
        if device:
            self.model.to(device)

    def infer(
            self,
            frame: np.ndarray,
            confidence: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> sv.Detections:
        """
        Запустить детекцию на одном кадре.

        Returns:
            Detections — уже отфильтрованные по уровню уверенности
        """
        result = self.model(frame)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()

        det = sv.Detections(
            xyxy=boxes,
            class_id=class_ids,
            confidence=scores
        )
        return det[det.confidence >= confidence]
