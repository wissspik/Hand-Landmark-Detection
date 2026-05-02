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

Для обучения на видеокарте нужна CUDA-сборка PyTorch:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --timeout 120 --retries 10
```

Проверка CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Если `torch.cuda.is_available()` выводит `False`, обучение на GPU не запустится.

## Обучение

Основной запуск:

```powershell
python .\train_keypoints.py --pretrain-epochs 7 --finetune-epochs 30 --batch-size 16 --save-path .\best_keypoint_model_new.pt
```

Что происходит:

1. `pretrain`: обучение на `images/train`, проверка на `images/val`.
2. Загружается лучший state после pretrain.
3. `finetune`: дообучение на `images/train_my`, проверка на `images/val_my`.
4. В `best_keypoint_model_new.pt` сохраняется лучший чекпоинт по `val_my`, а не последняя эпоха.
5. Рядом сохраняются метрики и график:
   - `best_keypoint_model_new.metrics.json`
   - `best_keypoint_model_new.metrics.png`

Быстрая проверка механики:

```powershell
python .\train_keypoints.py --pretrain-epochs 1 --finetune-epochs 1 --batch-size 16 --save-path .\best_keypoint_model_new.pt
```

Полезные параметры:

- `--pretrain-epochs` - количество эпох на внешнем датасете `images/train`.
- `--finetune-epochs` - количество эпох дообучения на твоих webcam-данных.
- `--batch-size` - размер батча. Если видеокарта нестабильна или не хватает VRAM, уменьшить до `16` или `8`.
- `--save-path` - куда сохранить лучший `.pt`.
- `--crop-margin` - запас вокруг руки при crop. По умолчанию `0.25`.

## Запуск на веб-камере

```powershell
python .\run_webcam_keypoints.py --checkpoint .\best_keypoint_model_new.pt --camera-id 2
```

Если не знаешь индекс камеры, можно не указывать `--camera-id`: скрипт попробует найти рабочую камеру сам.

```powershell
python .\run_webcam_keypoints.py --checkpoint .\best_keypoint_model_new.pt
```

Закрыть окно: `q` или `Esc`.

Запуск на CPU:

```powershell
python .\run_webcam_keypoints.py --checkpoint .\best_keypoint_model_new.pt --use-cpu
```

Проверка на отдельных изображениях:

```powershell
python .\run_webcam_keypoints.py --checkpoint .\best_keypoint_model_new.pt --image-paths images\val_my\frame_000000.jpg --output-dir inference_outputs
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

Схема:

```text
webcam frame
  -> MediaPipe hand detection
  -> bbox
  -> crop hand
  -> letterbox 224x224
  -> CNN
  -> predicted 21 keypoints
  -> unletterbox
  -> uncrop
  -> draw on original frame
```

## Train / Val / Test

В этом проекте используются `train` и `val`.

- `train` - данные, на которых модель реально учится.
- `val` - данные, на которых модель проверяется после эпохи. На них веса не обновляются.

Внешний датасет:

- `images/train` - обучение на общем датасете рук;
- `images/val` - проверка качества на общем датасете рук.

Твой webcam-датасет:

- `images/train_my` - дообучение под твою вебку;
- `images/val_my` - проверка под реальный сценарий.

Финальный `.pt` выбирается по `val_my`, потому что задача проекта - хорошо работать на веб-камере.

## Рекомендованные настройки

Для локального обучения на нестабильной Windows/GPU:

```powershell
python .\train_keypoints.py --pretrain-epochs 7 --finetune-epochs 30 --batch-size 16 --save-path .\best_keypoint_model_new.pt
```

Для Colab или более стабильной GPU:

```powershell
python .\train_keypoints.py --pretrain-epochs 7 --finetune-epochs 30 --batch-size 64 --save-path .\best_keypoint_model_new.pt
```

Если качество на вебке плохое, чаще всего надо не просто увеличивать эпохи, а добавлять больше своих webcam-кадров: разные расстояния до камеры, освещение, повороты ладони, жесты и фон.
