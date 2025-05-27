"""
SPC — геометрия стандартного футбольного поля:
размеры, вершины, рёбра для отрисовки линий
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing      import List, Tuple


@dataclass(slots=True)
class SoccerPitchConfiguration:
    """Параметры 105×68 м (7000 × 12000 см по оси y × x)."""

    # --------------------- размеры --------------------- #
    width:  int = 7000     # см  (ось Y)
    length: int = 12000    # см  (ось X)

    penalty_box_width:  int = 4100
    penalty_box_length: int = 2015
    goal_box_width:     int = 1832
    goal_box_length:    int = 550
    centre_circle_radius:   int = 915
    penalty_spot_distance: int = 1100

    # --------------------- вершины --------------------- #
    @property
    def vertices(self) -> List[Tuple[int, int]]:
        w, l = self.width, self.length
        PBW, PBL = self.penalty_box_width, self.penalty_box_length
        GBW, GBL = self.goal_box_width,    self.goal_box_length
        CCR, PSD = self.centre_circle_radius, self.penalty_spot_distance
        # (x, y)
        return [
            # левая сторона
            (0, 0),                          # 01
            (0, (w-PBW)//2),                 # 02
            (0, (w-GBW)//2),                 # 03
            (0, (w+GBW)//2),                 # 04
            (0, (w+PBW)//2),                 # 05
            (0, w),                          # 06
            # ворота + штрафная слева
            (GBL, (w-GBW)//2),               # 07
            (GBL, (w+GBW)//2),               # 08
            (PSD, w//2),                     # 09   penalty spot
            (PBL, (w-PBW)//2),               # 10
            (PBL, (w-GBW)//2),               # 11
            (PBL, (w+GBW)//2),               # 12
            (PBL, (w+PBW)//2),               # 13
            # центральная линия и круг
            (l//2, 0),                       # 14
            (l//2, w//2-CCR),                # 15
            (l//2, w//2+CCR),                # 16
            (l//2, w),                       # 17
            # правая штрафная (зеркально)
            (l-PBL, (w-PBW)//2),             # 18
            (l-PBL, (w-GBW)//2),             # 19
            (l-PBL, (w+GBW)//2),             # 20
            (l-PBL, (w+PBW)//2),             # 21
            (l-PSD, w//2),                   # 22
            (l-GBL, (w-GBW)//2),             # 23
            (l-GBL, (w+GBW)//2),             # 24
            # правый край
            (l, 0),                          # 25
            (l, (w-PBW)//2),                 # 26
            (l, (w-GBW)//2),                 # 27
            (l, (w+GBW)//2),                 # 28
            (l, (w+PBW)//2),                 # 29
            (l, w),                          # 30
            # доп. точки для круга
            (l//2-CCR, w//2),                # 31
            (l//2+CCR, w//2),                # 32
        ]

    # --------------------- рёбра --------------------- #
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1,2),(2,3),(3,4),(4,5),(5,6),      # левая бровка
        (7,8),                              # ворота
        (10,11),(11,12),(12,13),            # левая штрафная
        (14,15),(15,16),(16,17),            # центр
        (18,19),(19,20),(20,21),            # правая штрафная
        (23,24),                            # ворота
        (25,26),(26,27),(27,28),(28,29),(29,30),   # правая бровка
        # вертикали
        (1,14),(14,25),
        # доп. связи
        (2,10),(3,7),(4,8),(5,13),(6,17),
        (21,29),(17,30)
    ])