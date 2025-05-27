"""
futai.detector
==============

Здесь живёт единая «фабрика» детекторов.
Сейчас factory оборачивает Ultralytics-YOLO,
при желании можно добавить OpenVINO/ONNX/SAM и т.п.
"""
from __future__ import annotations
from pathlib import Path
from typing import Protocol, Any
from ultralytics import YOLO


class Detector(Protocol):
    """Минимальный API, которое ждёт конвейер."""
    def infer(self, frame: Any, *, confidence: float) -> Any: ...


class YOLODetector:                       # <— раньше это был PlayerDetectionModel
    """Обёртка над Ultralytics-YOLO с единым методом infer."""
    def __init__(self, weights: str | Path, device: str | None = None) -> None:
        self.model = YOLO(str(weights))
        if device:
            self.model.to(device)

    def infer(self, frame, *, confidence: float = 0.3, **kwargs):
        """Возвращает raw-`Results` Ultralytics."""
        return self.model(frame, conf=confidence, verbose=False, **kwargs)


def build_detector(kind: str, weights: str, device: str | None = None) -> Detector:
    """
    Factory.  Параметр *kind* позволяет в будущем легко подменять backend.
    """
    kind = kind.lower()
    if kind in {"yolo", "yolov8", "ultralytics"}:
        return YOLODetector(weights, device)
    raise ValueError(f"Unknown detector type: {kind!r}")