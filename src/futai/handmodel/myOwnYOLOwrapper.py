from __future__ import annotations
from ultralytics import YOLO


class LocalYOLOWrapper:
    def __init__(self, weights_path: str, device: str = "cuda") -> None:
        self.model = YOLO(weights_path)
        self.device = device

    def infer(self, img, confidence: float = 0.3, **kwargs):
        """
        Инференс

        Параметры Ultralytics, см по ссылке:
        https://docs.ultralytics.com/ru/modes/predict/#inference-arguments
        тут будет калька
        img : np.ndarray / str / pathlib.Path
            Изображение либо путь к изображению/видео.
        confidence
            Нижняя граница score для вывода объектов
        **kwargs
            Любые доп аргументы Ultralytics-YOLO (iou, classes).

        вернет List[ultralytics.engine.results.Results]
        """
        return self.model(img, conf=confidence, verbose=False, **kwargs)
