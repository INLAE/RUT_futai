"""
Утилиты — отдельно вынесена логика назначения вратарей к команде.
"""

import numpy as np
import supervision as sv

from .constants import GK_CLASS_ID, PLAYER_CLASS_ID

def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Простая эвристика: берём bottom-center игроков обеих команд,
    вычисляем их центроиды и атрибутируем каждого вратаря
    к ближайшему centroid.
    """
    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pl_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    cent0 = pl_xy[players.class_id == 0].mean(axis=0)
    cent1 = pl_xy[players.class_id == 1].mean(axis=0)

    result = []
    for xy in gk_xy:
        d0 = np.linalg.norm(xy - cent0)
        d1 = np.linalg.norm(xy - cent1)
        result.append(0 if d0 < d1 else 1)

    return np.array(result, dtype=int)