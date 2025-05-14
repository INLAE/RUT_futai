# -*- coding: utf-8 -*-

"""
Главный конвейер: от кадра до аннотированного output’а.
"""

import supervision as sv
import numpy as np

from .detection      import PlayerDetectionModel
from .classification import TeamClassifier
from .tracking       import Tracker
from .annotation     import build_annotators
from .utils          import resolve_goalkeepers_team_id
from .constants      import (
    BALL_CLASS_ID,
    GK_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    DEFAULT_CONFIDENCE_THRESHOLD
)

class TeamVideoProcessor:
    """
    Pipeline:
      1) Детекция (YOLO)
      2) Трекинг (ByteTrack)
      3) Классификация команд (SigLIP→UMAP→KMeans)
      4) Assign GK once
      5) Аннотации (ellipse, label, triangle)
    """

    def __init__(
        self,
        weights_path: str,
        video_path:   str,
        device:       str = None,
        confidence:   float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        # Детектор + трекер + классификатор
        self.detector      = PlayerDetectionModel(weights_path, device)
        self.tracker       = Tracker()
        self.team_clf      = TeamClassifier(device=device)
        # Генератор фреймов
        self.frame_gen     = sv.get_video_frames_generator(video_path)
        self.confidence    = confidence
        # Мапа GK tracker_id → команда (0/1)
        self.gk_team_map   : dict[int,int] = {}
        # Annotators
        self.annotators    = build_annotators()

    def process_next(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Обработать следующий фрейм:
          - детектим, трекаем, классифицируем, аннотируем.
        Возвращаем (оригинал, аннотированный).
        """
        frame = next(self.frame_gen)

        # 1) Детекция
        dets = self.detector.infer(frame, confidence=self.confidence)

        # 2) Отделить мяч и pad
        ball = dets[dets.class_id == BALL_CLASS_ID]
        ball.xyxy = sv.pad_boxes(ball.xyxy, px=10)

        # 3) Трекинг остальных
        others = dets[dets.class_id != BALL_CLASS_ID]
        others = others.with_nms(threshold=0.5, class_agnostic=True)
        others = self.tracker.update(others)

        # 4) Разделить по классам
        gk      = others[others.class_id == GK_CLASS_ID]
        players = others[others.class_id == PLAYER_CLASS_ID]
        refs    = others[others.class_id == REFEREE_CLASS_ID]

        # 5) Классификация команд для игроков
        crops = [sv.crop_image(frame, xy) for xy in players.xyxy]
        players.class_id = self.team_clf.predict(crops)

        # 6) Назначить GK один раз и зафризить цвет
        for tid in gk.tracker_id:
            if tid not in self.gk_team_map:
                # вызывать утилиту по одному элементу
                self.gk_team_map[tid] = int(
                    resolve_goalkeepers_team_id(
                        players, gk[gk.tracker_id == tid]
                    )[0]
                )
            # присвоить ему запомненную команду
            gk.class_id[gk.tracker_id == tid] = self.gk_team_map[tid]

        # 7) Сдвинуть referee ID в zero-based
        refs.class_id -= 1

        # 8) Merge & Annotate
        merged = sv.Detections.merge([players, gk, refs])
        merged.class_id = merged.class_id.astype(int)

        ann = frame.copy()
        ann = self.annotators['ellipse'].annotate(scene=ann, detections=merged)
        labels = [f"#{tid}" for tid in merged.tracker_id]
        ann = self.annotators['label'].annotate(scene=ann, detections=merged, labels=labels)
        ann = self.annotators['triangle'].annotate(scene=ann, detections=ball)

        return frame, ann