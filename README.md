# Bubble YOLO

Обучение и преразметка пузырей (`rising` / `sitting`) через Ultralytics YOLO.  
Разметка для обучения — в `data/dataset_*_clean/`.

Артефакты (dataset, runs, веса) лежат локально в `yolo/` и в git не попадают.

---

## Установка

```powershell
cd ...\Bubbles
pip install -e .
```

---

## 1) Собрать train/val

```powershell
python -m bubble_yolo prepare --all-clean
```

Или выборочно (короткое имя → `data/dataset_<имя>_clean`):

```powershell
python -m bubble_yolo prepare 12 13 x1 x2
```

Результат: `yolo/dataset/` + `data.yaml`.

---

## 2) Обучить

```powershell
python -m bubble_yolo train
```

Полезные флаги:

```powershell
python -m bubble_yolo train --epochs 200 --batch 4 --device cpu
python -m bubble_yolo train --model yolov8s.pt
```

Веса: `yolo/runs/bubbles/weights/best.pt`

---

## 3) Преразметить новый датасет

```powershell
python -m bubble_yolo predict --source data/dataset_NEW/images --save-overlay --conf 0.25
```

Пишет:

- `data/dataset_NEW/labels_pred/` — YOLO `.txt`
- `data/dataset_NEW/labels_pred_overlay/` — превью с боксами

Потом правь labels вручную / в Label Studio и сохраняй как `dataset_NEW_clean`.

---

## Заметки

- Кропы узкие (~80×640) → в train включены `rect=True` и аккуратный аугмент.
- Чем больше `*_clean`, тем лучше; цель — меньше ручной разметки, не идеальный детектор.
