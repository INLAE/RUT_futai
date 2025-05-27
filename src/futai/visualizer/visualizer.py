"""
Visualizer — вывод кадра / радара / диаграмм Вороного.
"""
from __future__ import annotations
import numpy as np, supervision as sv
from ..pitch.config import SoccerPitchConfiguration as CFG
from ..pitch.draw import PitchDrawer as PD


class Visualizer:
    def __init__(self, ellipse: sv.EllipseAnnotator,
                 triangle: sv.TriangleAnnotator,
                 label: sv.LabelAnnotator):
        self.e, self.t, self.l = ellipse, triangle, label
        self.cfg = CFG()

    # исходный кадр
    def frame(self, frame, ball, others):
        img = frame.copy()
        img = self.t.annotate(img, ball)
        img = self.e.annotate(img, others)
        img = self.l.annotate(img, others,
                              [f"#{tid}" for tid in others.tracker_id])
        sv.plot_image(img)

    # радар
    def radar(self, ball_xy, pl_xy, team_flag, ref_xy=np.empty((0, 2))):
        img = PD.draw_pitch(self.cfg)
        img = PD.draw_points_on_pitch(self.cfg, ball_xy, sv.Color.WHITE,
                                      sv.Color.BLACK, 10, pitch=img)
        img = PD.draw_points_on_pitch(self.cfg, pl_xy[team_flag == 0],
                                      sv.Color.from_hex("00BFFF"), sv.Color.BLACK, 16, pitch=img)
        img = PD.draw_points_on_pitch(self.cfg, pl_xy[team_flag == 1],
                                      sv.Color.from_hex("FF1493"), sv.Color.BLACK, 16, pitch=img)
        sv.plot_image(img)

    # blend диаграммы Воронного + точки
    def voronoi_blend(self, pl_xy, team_flag, opacity=0.45):
        img = PD.draw_pitch(self.cfg, background=sv.Color.WHITE,
                            line_color=sv.Color.BLACK)
        img = PD.draw_pitch_voronoi_diagram(self.cfg,
                                            pl_xy[team_flag == 0],
                                            pl_xy[team_flag == 1],
                                            sv.Color.from_hex("00BFFF"),
                                            sv.Color.from_hex("FF1493"),
                                            opacity, pitch=img)
        img = PD.draw_points_on_pitch(self.cfg, pl_xy[team_flag == 0],
                                      sv.Color.from_hex("00BFFF"),
                                      sv.Color.WHITE, 16, 1, img)
        img = PD.draw_points_on_pitch(self.cfg, pl_xy[team_flag == 1],
                                      sv.Color.from_hex("FF1493"),
                                      sv.Color.WHITE, 16, 1, img)
        sv.plot_image(img)
