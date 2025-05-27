# RUT-FUT-AI — трекинг футболистов, радар и диаграммы Вороного

**RUT-FUT-AI** — мой дипломный репозиторий, позволяющий из одного или нескольких видео-кадров получить:

* детектированные и отслеживаемые объекты  
  (мяч / игроки / вратари / судья) с идентификаторами трека;
* автоматическое разбиение игроков на 2 команды без разметки;
* радар — упрощенный план поля сверху, как в игре FIFA;
* контрольные зоны команд (Диаграмма Воронного) поверх радара;
* два готовых видеофайла: *detect_out.mp4* (кадр + боксы) и *radars_out.mp4*.

Библиотеки: **Ultralytics YOLOv8 + Supervision + SigLIP + UMAP + scikit-learn**.  
Код разбит на небольшие, независимые модули (см. `src/futai/*`).

## Пререквизиты
> [!TIP]
> Перед применением сервиса, необходимо иметь веса обученной модели для детекции игроков и 32-х точек на поле. В данном репозитории ноутбуки с обучением этих моделей располагаются в директории `JupyterNotebooks`.  

---

## 1. Установка

```bash
git clone https://github.com/your-name/rut-fut-ai.git
cd rut-fut-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> [!IMPORTANT]  
> YOLO и SigLIP по-умолчанию используют GPU. Если запускаете на CPU, измените DEFAULT_DEVICE в src/futai/constants.py.

## 2. Структура проекта
````
src/futai
├── detector/        YOLO детекция и ByteTrack
├── handmodel/       классификатор команд (SigLIP > UMAP > KMeans)
├── pitch/           геометрия поля + функции рисования
├── utils/           утилиты (назначение GK к команде и тд)
└── visualizer/      вывод кадра / радара / диаграммы Воронного
````
## 3. Использование
```python
from pathlib import Path
import supervision as sv
import cv2, numpy as np

# 1. Пути к данным
VIDEO_IN Path("/content/CSKA.mp4") # исходное видео
PLAYER_WEIGHTS = Path("/content/weights/players.pt") # YOLO модель игроков
FIELD_WEIGHTS = Path("/content/weights/field.pt") # YOLO модель линий поля

# 2. Инициализация пайплайна
from futai.detector.detection import PlayerDetectionModel
from futai.handmodel.classification import TeamClassifier
from futai.detector.tracking import Tracker
from futai.utils.gk_resolver import GoalkeeperResolver as GK
from futai.pitch.config import SoccerPitchConfiguration as SPC
from futai.pitch.pitch_projector import PitchProjector
from futai.visualizer.visualizer import Visualizer
from futai.detector.annotation import build_annotators

player_det = PlayerDetectionModel(str(PLAYER_WEIGHTS))
team_clf = TeamClassifier()
tracker = Tracker()
pitch_proj = PitchProjector(
    field_model = PlayerDetectionModel(str(FIELD_WEIGHTS))
)  # использует тот же YOLO-wrapper
vis = Visualizer(**build_annotators())

# 3. Читаем видео
cap = cv2.VideoCapture(str(VIDEO_IN))
ret, frame = cap.read()    # <— для примера возьмём первый кадр

# 4. Шаги конвейера
## 4-A. детекция + трекинг
ball_det, other_det = player_det.infer(frame).split_ball_and_others(tracker)

## 4-B. классификация команд
crops = [sv.crop_image(frame, xy) for xy in other_det.players.xyxy]
other_det.players.class_id = team_clf.predict(crops)
other_det.goalkeepers.class_id = GK.resolve(other_det.players,
                                            other_det.goalkeepers)

## 4-C. аннотированный кадр
vis.frame(frame, ball_det, other_det.all)

## 4-D. проекция на птичий взгляд
proj = pitch_proj.project(
    frame,
    {
        "ball"  : ball_det.bottom_center(),
        "player": other_det.all.bottom_center(),
    }
)
ball_xy_p   = proj["ball"]
player_xy_p = proj["player"]
team_flag   = other_det.all.class_id

## 4-E. рисуем радар и диаграмму Воронного
vis.radar(ball_xy_p, player_xy_p, team_flag)
vis.voronoi_blend(player_xy_p, team_flag, opacity=0.45)
```

> [!NOTE]
> Классификация команд полностью unsupervised — SigLIP + UMAP + KMeans.
При смене формы достаточно 50–100 кадров для ре-фита.
> Модуль pitch не зависит от ML — его можно использовать отдельно
для любых визуализаций на плоскости поля.
