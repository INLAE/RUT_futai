"""
PitchProjector — безопасная гомография frame -> модель поля.
"""
from __future__ import annotations
import cv2, numpy as np, supervision as sv
from src.futai.pitch.config import SoccerPitchConfiguration as CFG


class PitchProjector:
    def __init__(self, field_model):
        self.field_model = field_model

    def _homography(self, src, dst):
        return None if len(src) < 4 else cv2.findHomography(src, dst, 0)[0]
    def project(self, frame, points: dict[str, np.ndarray]):
        res = self.field_model.infer(frame, conf=0.3)[0]
        kpts = sv.KeyPoints.from_ultralytics(res)
        good = kpts.confidence[0] > .5
        H = self._homography(kpts.xy[0][good], np.asarray(CFG().vertices)[good])

        out = {}
        for k, v in points.items():
            if H is None or len(v) == 0:
                out[k] = np.empty((0, 2))
            else:
                out[k] = cv2.perspectiveTransform(
                    v.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
        return out
