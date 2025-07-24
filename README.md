# RUT-FUT-AI — трекинг футболистов, радар и диаграммы Вороного

**RUT-FUT-AI** — мой дипломный репозиторий, позволяющий из одного или нескольких видео-кадров получить:

* детектированные и отслеживаемые объекты  
  (мяч / игроки / вратари / судья) с идентификаторами трека;
* автоматическое разбиение игроков на 2 команды без разметки;
* радар — упрощенный план поля сверху, как в игре FIFA;
* контрольные зоны команд (Диаграмма Воронного) поверх радара;
* два готовых видеофайла: *detect_out.mp4* (кадр + боксы) и *radars_out.mp4*.

Библиотеки: **Ultralytics YOLOv8 + Supervision + SigLIP + UMAP + scikit-learn**.  

[Демонстрация](/data/demo.mp4)

<video src="https://raw.githubusercontent.com/INLAE/RUT_futai/main/data/demo.mp4" width="640" controls></video>
       
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
# Ячейка 1. Библиотеки
from pathlib import Path  # для работы с путями к файлам
import cv2, numpy as np
import supervision as sv  # надстройка-визуализатор для YOLO

# Ячейка 2. Пути на источники весов и видео
VIDEO_IN = Path("/content/Rotor.mp4")  # исходный матч
PLAYER_WEIGHTS = Path("/content/weights/players.pt")  # веса YOLO для игроков
FIELD_WEIGHTS = Path("/content/weights/field.pt")  # веса YOLO для линий поля

# Ячейка 3. Инициализация пайпа
from futai.detector.detection import PlayerDetectionModel
from futai.handmodel.classification import TeamClassifier
from futai.detector.tracking import Tracker
from futai.utils.gk_resolver import GoalkeeperResolver as GK
from futai.pitch.config import SoccerPitchConfiguration as SPC
from futai.pitch.pitch_projector import PitchProjector
from futai.visualizer.visualizer import Visualizer
from futai.detector.annotation import build_annotators

# Ячейка 4. Модели
player_det = PlayerDetectionModel(str(PLAYER_WEIGHTS))  # YOLO для игроков
team_clf = TeamClassifier()  # Siglip + UMAP + KMeans
tracker = Tracker()  # ByteTrack
pitch_proj = PitchProjector(  # YOLO для линий поля
  field_model=PlayerDetectionModel(str(FIELD_WEIGHTS))
)
vis = Visualizer(**build_annotators())  # карандаши supervision

# Ячейка 5. Берем любой кадр из видео
cap = cv2.VideoCapture(str(VIDEO_IN))
ret, frame = cap.read()  # для демо достаточно первого полученного кадра

# Ячейка 6. Детекция + трекинг
# метод split_ball_and_others — это условная обкртка, которая возвращает отдельно мяч (ball_det) и стальных» (игроки, гк, судья), уже с трекингом
ball_det, other_det = player_det.infer(frame).split_ball_and_others(tracker)

# Ячейка 7. Определяем, кто в какой команде, обрезая квадратиками всех игроков -> скармливаем классификатору
crops = [sv.crop_image(frame, xy) for xy in other_det.players.xyxy]
other_det.players.class_id = team_clf.predict(crops)

# для вратарей используем простую эвристику ближайший к центру своей команды
other_det.goalkeepers.class_id = GK.resolve(other_det.players,
                                            other_det.goalkeepers)

# Ячейка 8. Рисуем одиночный кадр
vis.frame(  # добавит боксы, ellipse контуры и подписи
  frame,  # оригинальный RGB кадр
  ball_det,  # отдельная детекция мяча
  other_det.all  # все остальные объекты вместе
)

# Ячейка 9. Птичий взгляд (координаты на схеме поля)
proj = pitch_proj.project(  # строим гомографию кадр -> модель поля
  frame,
  {
    "ball": ball_det.bottom_center(),  # точка мяча
    "player": other_det.all.bottom_center(),  # точки игроков/гк
  }
)
ball_xy_p = proj["ball"]  # мяч на схеме (x, y в см)
player_xy_p = proj["player"]  # игроки на схеме
team_flag = other_det.all.class_id  # 0 или 1 — цвет каждой точки

# Ячейка 10. Радар и диаграмма Вороного
vis.radar(  # маленькая мини-карта как в фифе
  ball_xy_p,
  player_xy_p,
  team_flag
)
vis.voronoi_blend(  # раскрашенная диаграмма - зона контроля
  player_xy_p,
  team_flag,
  opacity=0.45
)

# 3. Читаем видео
cap = cv2.VideoCapture(str(VIDEO_IN))
ret, frame = cap.read()  # <— для примера возьмём первый кадр

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
    "ball": ball_det.bottom_center(),
    "player": other_det.all.bottom_center(),
  }
)
ball_xy_p = proj["ball"]
player_xy_p = proj["player"]
team_flag = other_det.all.class_id

## 4-E. рисуем радар и диаграмму Воронного
vis.radar(ball_xy_p, player_xy_p, team_flag)
vis.voronoi_blend(player_xy_p, team_flag, opacity=0.45)
```

> [!NOTE]
> Классификация команд полностью unsupervised — SigLIP + UMAP + KMeans.
При смене формы достаточно 50–100 кадров для ре-фита.
> Модуль pitch не зависит от ML — его можно использовать отдельно
для любых визуализаций на плоскости поля.
