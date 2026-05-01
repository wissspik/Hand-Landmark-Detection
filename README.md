# Hand Keypoint Dataset 26k

Проект для сбора датасета рук, подготовки аугментаций, обучения простой модели ключевых точек и проверки результата на веб-камере или отдельных изображениях.

## Что внутри

- `capture_mediapipe_dataset.py` - запись кадров с веб-камеры и сохранение разметки MediaPipe в JSONL.
- `augment_train_my_dataset.py` - аугментация собственного датасета `images/train_my`.
- `augment_coco_unusual.py` - генерация "unusual" COCO-разметки и изображений на основе `coco_annotation/train` и `coco_annotation/val`.
- `coco_torch_dataset.py` - PyTorch Dataset для COCO-аннотаций.
- `train_keypoints_one_file.py` - обучение CNN-модели для 21 ключевой точки руки.
- `run_webcam_keypoints.py` - inference через веб-камеру или список изображений.
- `images/`, `labels/`, `val/`, `coco_annotation/` - данные и аннотации.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy opencv-python mediapipe pillow torch matplotlib
```

## Быстрый старт

Сбор кадров и разметки:

```powershell
python capture_mediapipe_dataset.py --hand-landmarker-model hand_landmarker.task --interval-sec 1
```

Аугментация собственного датасета:

```powershell
python augment_train_my_dataset.py --src-images-dir images/train_my --src-annotations images/train_my.annotations.jsonl --dst-images-dir images/train_my_aug --dst-annotations images/train_my_aug.annotations.jsonl
```

Обучение модели:

```powershell
python train_keypoints_one_file.py --dataset-root . --epochs 20 --batch-size 64 --save-path best_keypoint_model.pt
```

Запуск на веб-камере:

```powershell
python run_webcam_keypoints.py --checkpoint best_keypoint_model.pt --hand-landmarker-model hand_landmarker.task
```

## Структура данных

Для обучения скрипт ожидает:

- `images/train_my/` и `images/train_my.annotations.jsonl`
- `images/val_my/` и `images/val_my.annotations.jsonl`

Каждая запись JSONL должна ссылаться на изображение и содержать координаты ключевых точек руки. COCO-часть проекта использует файлы `_annotations.coco.json` внутри `coco_annotation/<split>/`.
