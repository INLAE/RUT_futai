"""
Тут будут методы визуализации поля
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv
from typing import Optional, List
from .config import SoccerPitchConfiguration as CFG
from ..constants import FIELD_SCALE, FIELD_PADDING


class PitchDrawer:
    @staticmethod
    def _s(val: int, scale: float) -> int:
        # Конвертация сантиметры в пиксели
        return int(val * scale)

    # отрисовка чистого поля
    @staticmethod
    def draw_pitch(
            cfg: CFG,
            background: sv.Color = sv.Color(34, 139, 34),  # зелёный газон
            line_color: sv.Color = sv.Color.WHITE,  # линии белые
            padding: int = FIELD_PADDING,  # отступы по краям
            line_thickness: int = 4,  # толщина линий
            scale: float = FIELD_SCALE  # коэффициент масштаба
    ) -> np.ndarray:
        # создаём канву необходимого размера
        h = PitchDrawer._s(cfg.width, scale) + 2 * padding
        w = PitchDrawer._s(cfg.length, scale) + 2 * padding
        img = np.full((h, w, 3), background.as_bgr(), np.uint8)

        # проводим все линии
        for a, b in cfg.edges:
            p1 = (
                PitchDrawer._s(cfg.vertices[a - 1][0], scale) + padding,
                PitchDrawer._s(cfg.vertices[a - 1][1], scale) + padding
            )
            p2 = (
                PitchDrawer._s(cfg.vertices[b - 1][0], scale) + padding,
                PitchDrawer._s(cfg.vertices[b - 1][1], scale) + padding
            )
            cv2.line(img, p1, p2, line_color.as_bgr(), line_thickness)

        # центр поля (круг + точка)
        cv2.circle(
            img,
            (w // 2, h // 2),
            PitchDrawer._s(cfg.centre_circle_radius, scale),
            line_color.as_bgr(),
            line_thickness
        )

        # две точки пенальти
        for x in (
                PitchDrawer._s(cfg.penalty_spot_distance, scale),
                w - PitchDrawer._s(cfg.penalty_spot_distance, scale)
        ):
            cv2.circle(img, (x, h // 2), 8, line_color.as_bgr(), -1)

        return img

    # выводим точки на поле сверху на уже нарисованное ставит кружочки-игроки/мяч/судью
    @staticmethod
    def draw_points_on_pitch(
            cfg: CFG,
            xy: np.ndarray,  # N*2 координат игроков/мяча
            face: sv.Color = sv.Color.RED,  # заливка
            edge: sv.Color = sv.Color.BLACK,  # обводка
            radius: int = 10,
            thickness: int = 2,
            padding: int = FIELD_PADDING,
            scale: float = FIELD_SCALE,
            pitch: Optional[np.ndarray] = None  # можно передать готовое поле
    ) -> np.ndarray:
        if pitch is None:
            # если нет — рисуем новое
            pitch = PitchDrawer.draw_pitch(cfg, padding=padding, scale=scale)

        # рисуем каждую точку
        for x, y in xy:
            pt = (
                PitchDrawer._s(x, scale) + padding,
                PitchDrawer._s(y, scale) + padding
            )
            cv2.circle(pitch, pt, radius, face.as_bgr(), -1)  # заливка
            cv2.circle(pitch, pt, radius, edge.as_bgr(), thickness)  # контур
        return pitch

    # Диаграмма Вороного
    # Исходя из логики: чья территория ближе к этому пикселю для обеих команд
    @staticmethod
    def draw_pitch_voronoi_diagram(
            cfg: CFG,
            team1_xy: np.ndarray,  # координаты команды 1
            team2_xy: np.ndarray,  # координаты команды 2
            team1_color: sv.Color = sv.Color.RED,
            team2_color: sv.Color = sv.Color.WHITE,
            opacity: float = 0.5,
            padding: int = FIELD_PADDING,
            scale: float = FIELD_SCALE,
            pitch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if pitch is None:
            pitch = PitchDrawer.draw_pitch(cfg, padding=padding, scale=scale)

        # сеточка всех пикселей изображения
        h, w, _ = pitch.shape
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        grid_y -= padding
        grid_x -= padding

        # функция для определения мин дистанции от пикселя до ближайшего игрока
        def _dist(pts: np.ndarray) -> np.ndarray:
            return np.min(
                (
                        (PitchDrawer._s(pts[:, 0][:, None, None], scale) - grid_x) ** 2 +
                        (PitchDrawer._s(pts[:, 1][:, None, None], scale) - grid_y) ** 2
                ),
                axis=0
            )

        # где ближе игроки первой / второй команды
        mask = _dist(team1_xy) < _dist(team2_xy)

        # готовим двухцветную заливку
        color1 = np.array(team1_color.as_bgr(), np.uint8)
        color2 = np.array(team2_color.as_bgr(), np.uint8)
        overlay = np.where(mask[..., None], color1, color2)

        # прозрачный бленд поверх поля
        return cv2.addWeighted(overlay, opacity, pitch, 1 - opacity, 0)
