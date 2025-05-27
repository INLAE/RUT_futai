from __future__ import annotations
from __future__ import annotations

import numpy as np, supervision as sv

"""
Простая эвристика: берём bottom-center игроков обеих команд,
вычисляем их центроиды и атрибутируем каждого вратаря
к ближайшему centroid. Ближайший центр это и есть его команда
"""

from typing import Final
import numpy as np
import supervision as sv


class GoalkeeperResolver:
    @staticmethod
    def resolve(players: sv.Detections,
                goalkeepers: sv.Detections) -> np.ndarray:
        """
        NOTE!
        Параметры:
        players: Detections  – опознанные игроки с уже заполненным .class_id (0/1)
        goalkeepers: Detections  – боксы вратарей (class_id пока неважен)
        Требуется:
        массив[int] того же размера, что len(goalkeepers):
            0 -> первая команда, 1 -> вторая команда
        """

        # возьмем опорную точку бокса в его низу по центру - будет позицией игрока
        # просто вынесем константу
        POS = sv.Position.BOTTOM_CENTER
        # 1] координаты bottom-center всех объектов
        pl_xy = players.get_anchors_coordinates(POS)  # shape (Npl, 2)
        gk_xy = goalkeepers.get_anchors_coordinates(POS)  # shape (Ngk, 2)

        # 2] центроиды для каждой команды
        # Коль вдруг на кадре нет игроков одной из команд,
        # вычислять среднее нельзя -> используем NaN и разрулим позже
        cent_team0 = pl_xy[players.class_id == 0].mean(axis=0, keepdims=True)
        cent_team1 = pl_xy[players.class_id == 1].mean(axis=0, keepdims=True)

        # 3] расстояние до центров и выбор ближайшего
        dist_to_0 = np.linalg.norm(gk_xy - cent_team0, axis=1)
        dist_to_1 = np.linalg.norm(gk_xy - cent_team1, axis=1)

        # если центроид отсутствует (NaN) ->> дистанция = +inf
        # Если на кадре нет игроков какой-то команды, mean вернет NaN — меняем расстояние на +inf, вратарь автоматически
        # прикрепится к другой команде (у которой есть игроки)
        dist_to_0 = np.where(np.isnan(dist_to_0), np.inf, dist_to_0)
        dist_to_1 = np.where(np.isnan(dist_to_1), np.inf, dist_to_1)

        team_ids = (dist_to_1 < dist_to_0).astype(int)  # True = 1, False =0
        return team_ids
