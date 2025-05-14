# -*- coding: utf-8 -*-

"""
Обёртка над YOLO для детекции мяча, игроков и судьи.
"""

import numpy as np
import supervision as sv
from ultralytics import YOLO

from .constants import DEFAULT_CONFIDENCE_THRESHOLD

class PlayerDetectionModel:
    """
    Класс-обёртка для YOLOv8.
    """

    def __init__(self, weights_path: str, device: str = None):
        """
        Args:
            weights_path: путь к файлу весов .pt
            device: 'cpu' или 'cuda'; если None — YOLO выберет сам.
        """
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
            Detections — уже отфильтрованные по confidence.
        """
        result = self.model(frame)[0]
        boxes     = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        scores    = result.boxes.conf.cpu().numpy()

        det = sv.Detections(
            xyxy=boxes,
            class_id=class_ids,
            confidence=scores
        )
        return det[det.confidence >= confidence]