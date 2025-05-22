# Here We Go
# RUT-FUT Video Tracker

Инструмент для детекции, трекинга и классификации команд на спортивном видео с помощью:
- **YOLOv8** (детекция)
- **ByteTrack** (трекинг)
- **SigLIP → UMAP → KMeans** (кластеризация команд)
- **Supervision** (аннотации)

## Установка

```bash
git clone git@github.com:INLAE/RUT_futai.git
cd RUT_futai
pip install -e .
```
## Быстрый старт в Colab
```python
from google.colab import drive
drive.mount('/content/drive')

WEIGHTS   = '/content/drive/MyDrive/WEIGHTS/best.pt'
VIDEO_IN  = '/content/drive/MyDrive/VIDEO/video.mp4'
VIDEO_OUT = '/content/annotated.mp4'

from futai.processor import TeamVideoProcessor
import supervision as sv

processor = TeamVideoProcessor(
    weights_path=WEIGHTS,
    video_path=VIDEO_IN,
    device='cuda',
    confidence=0.3
)

# Пример одного кадра
frame, annotated = processor.process_next()
sv.plot_image(annotated)
```

## Полный прогон и сохранение
```python

import cv2

# Настройка VideoWriter
cap = cv2.VideoCapture(VIDEO_IN)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

# Сброс состояния
import supervision as sv
processor.frame_gen = sv.get_video_frames_generator(VIDEO_IN)
processor.tracker.reset()

# Обработка
while True:
    try:
        _, ann = processor.process_next()
    except StopIteration:
        break
    out.write(ann)

out.release()


```
## Важнецкая таблица для дебага и отладки
| Процесс                                            | Логика процесса                                           | Где найти в проекте?                         |
|----------------------------------------------------|-----------------------------------------------------------|----------------------------------------------|
| Загружаем обученную модель для поиска объектов     | import YOLOv8                                             | `PlayerDetectionModel` в `detection.py`      |
| Читаем видео по кадрам                             | `supervision.get_video_frames_generator()`                | в `TeamVideoProcessor` (`processor.py`)      |
| Детектируем игроков, мяч, вратарей и судей на кадре | YOLO `model(frame)` → `sv.Detections(...)`                | `infer()` в `detection.py`                   |
| Убираем лишние объекты, накладываем NMS            | `with_nms()` — убирает дублирующие боксы                  | в `process_next()`                           |
| Объединяем одних и тех же игроков в разных кадрах  | ByteTrack трекинг по координатам и размерам               | `tracker.update_with_detections()`           |
| Вырезаем изображения игроков из кадров             | `sv.crop_image()`                                         | `TeamClassifier.extract_features()`          |
| Переводим картинки игроков в векторы               | SigLIP (нейросеть) → UMAP → KMeans                        | `TeamClassifier` в `classification.py`       |
| Классифицируем игроков по командам без учителя     | Кластеризация (2 кластера)                                | `TeamClassifier.predict()`                   |
| Назначаем вратарей к ближайшей команде             | Эвристика — сравнение расстояний до центров масс игроков  | `resolve_goalkeepers_team_id()` в `utils.py` |
| Корректируем классы судей (нормализуем)            | `class_id -= 1`                                           | `process_next()`                             |
| Собираем все объекты обратно                       | `sv.Detections.merge([...])`                              | `process_next()`                             |
| Рисуем аннотации поверх кадра                      | `EllipseAnnotator`, `LabelAnnotator`, `TriangleAnnotator` | `annotation.py`                              |
| Возвращаем оригинальный и размеченный кадр         | tuple (original, annotated)                               | `process_next()` в `processor.py`            |
| Сохраняем кадры в видео файл                       | `cv2.VideoWriter(...).write(frame)`                       | В Colab-ноутбуке (финальный цикл записи)     |