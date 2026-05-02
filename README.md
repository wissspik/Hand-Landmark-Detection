# Hand Keypoint Dataset 26k

Проект для обучения и запуска модели, которая находит 21 ключевую точку руки на изображении или кадре с веб-камеры.

В проекте есть два типа данных:

- внешний COCO-датасет рук: `images/train`, `images/val`, `coco_annotation/train`, `coco_annotation/val`;
- собственные кадры с веб-камеры: `images/train_my`, `images/val_my`, `images/train_my.annotations.jsonl`, `images/val_my.annotations.jsonl`.

Идея обучения такая: сначала модель учится на большом внешнем датасете рук, потом дообучается на твоих кадрах с вебки. Финальный чекпоинт выбирается по качеству на `val_my`, потому что именно это ближе к реальному запуску.

## Структура проекта

- `train_keypoints.py` - обучение модели в две фазы: pretrain и fine-tune.
- `run_webcam_keypoints.py` - запуск модели на веб-камере или отдельных изображениях.
- `capture_mediapipe_dataset.py` - сбор кадров с веб-камеры и разметки через MediaPipe.
- `coco_torch_dataset.py` - вспомогательный Dataset для COCO-разметки.
- `images/train`, `images/val` - внешний датасет рук.
- `coco_annotation/train`, `coco_annotation/val` - COCO-аннотации для внешнего датасета.
- `images/train_my`, `images/val_my` - твои кадры с веб-камеры.
- `images/train_my.annotations.jsonl`, `images/val_my.annotations.jsonl` - разметка твоих кадров.
- `hand_landmarker.task` - модель MediaPipe для поиска руки.

## Установка

Создать и активировать виртуальное окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Базовые зависимости:

```powershell
python -m pip install numpy opencv-python mediapipe pillow matplotlib
```
## Обучение

Как происходит:

1. `pretrain`: обучение на `images/train`, проверка на `images/val`.
2. Загружается лучший state после pretrain.
3. `finetune`: дообучение на `images/train_my`, проверка на `images/val_my`.
4. В `best_keypoint_model_new.pt` сохраняется лучший чекпоинт по `val_my`.
5. Рядом сохраняются метрики и график:
   - `best_keypoint_model_new.metrics.json`
   - `best_keypoint_model_new.metrics.png`

## Запуск на веб-камере

```powershell
python run_webcam_keypoints.py --checkpoint best_keypoint_model_new.pt --camera-id 2
```

## Pipeline работы модели

### Во время обучения

1. Берется изображение и разметка 21 точки руки.
2. По размеченным точкам считается bounding box руки.
3. Вокруг bbox добавляется `crop-margin`.
4. Из исходного изображения вырезается crop руки.
5. Crop приводится к квадрату `224x224` через letterbox:
   - изображение масштабируется без искажения пропорций;
   - пустые области заполняются серым padding.
6. Координаты keypoints пересчитываются в систему координат `224x224`.
7. Координаты нормализуются в диапазон `[0, 1]`.
8. CNN получает картинку `3 x 224 x 224`.
9. CNN предсказывает `21 x 2` координаты keypoints.
10. Ошибка считается через `SmoothL1Loss` только по видимым точкам.
11. После каждой эпохи модель проверяется на validation split.
12. Лучший чекпоинт сохраняется по минимальному `val_loss`.

### Во время запуска на веб-камере

1. Скрипт получает кадр с веб-камеры.
2. MediaPipe ищет руку на кадре.
3. По точкам MediaPipe строится bbox руки.
4. Из кадра вырезается crop bbox.
5. Crop переводится в `224x224` через letterbox.
6. Crop подается в CNN.
7. CNN предсказывает 21 keypoint в координатах `224x224`.
8. Потом выполняется обратное преобразование координат:
   - координаты переводятся из `[0, 1]` в пиксели `224x224`;
   - убирается letterbox padding;
   - координаты масштабируются обратно в размер crop;
   - прибавляется смещение bbox на исходном кадре.
9. Получаются координаты keypoints уже в системе исходного кадра веб-камеры.
10. На экран рисуются bbox и скелет руки.

