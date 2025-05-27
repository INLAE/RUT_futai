"""
Главный конвейер: от кадра до аннотированного output’а.
"""

import supervision as sv
import numpy as np

from src.futai.detector.detection import PlayerDetectionModel
from src.futai.handmodel.classification import TeamClassifier
from src.futai.detector.tracking import Tracker
from src.futai.detector.annotation import build_annotators
from .constants import (
    BALL_CLASS_ID,
    GK_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    DEFAULT_CONFIDENCE_THRESHOLD  # он тут дефолтный!! 0.3 !
)
from .detector import build_detector


class TeamVideoProcessor:
    """
    Пайп:
      1) Детекция на YOLO
      2) Трекинг за ByteTrack
      3) Классификация команд SigLIP -> UMAP -> KMeans
      4) Assign GK once - эвристика закрепить первое ближайшее окружение
      5) Аннотации (ellipse, label, triangle)
    """

    def __init__(
            self,
            weights_path: str,
            video_path: str,
            detector_type: str = "yolo",
            device: str = None,
            confidence: float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        # Детектор + трекер + классификатор
        self.detector = build_detector(detector_type, weights_path, device)
        self.tracker = Tracker()
        self.team_clf = TeamClassifier(device=device)
        # Генератор фреймов
        self.frame_gen = sv.get_video_frames_generator(video_path)
        self.confidence = confidence
        # Мапа GK tracker_id -> команда (0/1)
        self.gk_team_map: dict[int, int] = {}
        # Annotators
        self.annotators = build_annotators()

    def process_next(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Обработать следующий фрейм:
          - детектим, трекаем, классифицируем, аннотируем.
        Возвращаем (оригинал, аннотированный).
        """
        # берем кадр и детектимё
        frame = next(self.frame_gen)

        # 1] Детекция
        res = self.detector.infer(frame, confidence=self.confidence)[0]
        dets = sv.Detections.from_ultralytics(res)

        # 2] делим на мяч и остальных (остальных трекаем)
        ball_det = dets[dets.class_id == BALL_CLASS_ID]
        ball_det.xyxy = sv.pad_boxes(ball_det.xyxy, px=10)  # чуть-чуть расширяем

        others = dets[dets.class_id != BALL_CLASS_ID]  # без мяча
        others = others.with_nms(0.5, class_agnostic=True)  # NMS
        others = self.tracker.update(others)  # ByteTrack

        # 3] раскладываем остальных по  своим ролям
        gk_det = others[others.class_id == GK_CLASS_ID]
        player_det = others[others.class_id == PLAYER_CLASS_ID]
        ref_det = others[others.class_id == REFEREE_CLASS_ID]

        # 4] классифицируем игроков по цвету формы (0 или 1)
        crops = [sv.crop_image(frame, xy) for xy in player_det.xyxy]
        player_det.class_id = self.team_clf.predict(crops)

        # 5] назначаем вратарей в ту же команду, что и ближайший кластер
        gk_det.class_id = GKRes.resolve(player_det, gk_det)

        # 6] referee: делаем class_id - 0 - команда A; или 1 - команда B
        ref_det.class_id -= 1  # было 3 -> станет 2 -> 1

        # 7] собираем всех вместе и рисуем
        all_det = sv.Detections.merge([player_det, gk_det, ref_det])
        all_det.class_id = all_det.class_id.astype(int)  # точно int32

        annotated = frame.copy()
        annotated = self.annotators["ellipse"].annotate(annotated, all_det)
        labels = [f"#{tid}" for tid in all_det.tracker_id]
        annotated = self.annotators["label"].annotate(annotated, all_det, labels)
        annotated = self.annotators["triangle"].annotate(annotated, ball_det)

        return frame, annotated
