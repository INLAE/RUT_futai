"""
ByteTrack-wrapper для сквозного трекинга объектов.
"""

import supervision as sv

class Tracker:
    """
    Обёртка вокруг sv.ByteTrack — даёт метод update().
    """

    def __init__(self):
        self._tracker = sv.ByteTrack()
        self.reset()

    def reset(self) -> None:
        """Сброс internal state трекера."""
        self._tracker.reset()

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Применить трекинг к входным детекциям.
        Возвращает Detections с полем tracker_id.
        """
        return self._tracker.update_with_detections(detections=detections)