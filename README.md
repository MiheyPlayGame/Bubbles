# Bubble YOLO

Пайплайн: **`.cine` → датасет → YOLO преразметка → правка → обучение**.

Классы: `0 = rising`, `1 = sitting`.

Артефакты обучения (`yolo/dataset`, `yolo/runs`, веса) в git не попадают.

---

## Установка

```powershell
cd ...\Bubbles
pip install -e .
```

---

## 0) Куда класть `.cine`

```text
data/films/
```

Пример: `data/films/24.cine`.

| Аргумент `source` | Что подхватится |
|---|---|
| `24` | `data/films/24.cine` |
| `24.cine` | `data/films/24.cine` |
| `data/films/24.cine` | этот путь |
| полный путь | как указано |

Работай из корня проекта (`...\Bubbles`).

---

## 1) Preview — линии ROI и enhance

Проверка `floor_y` / `rising_y` и параметров картинки на одном кадре:

```powershell
python -m bubble_yolo preview 24 --contrast 1.8 --brightness 30 --clahe
```

Результат в `data/preview_24/`:

- `original_guides.jpg` — линии на полном кадре  
- `enhanced_crop.png` — ROI после enhance  
- `preview.json` — геометрия + `image_params` для export

### Флаги preview

| Флаг | По умолчанию | Смысл |
|------|--------------|--------|
| `source` | — | stem / путь к `.cine` |
| `-o`, `--output` | `data/preview_<stem>` | папка результата |
| `--frame-index` | середина ролика | индекс кадра |
| `--floor-y` | авто | строка дна; должна быть **>** `rising_y` |
| `--rising-y` | `floor_y − crop_height` | верх ROI |
| `--side-crop-ratio` | `0.25` | **[0.0, 0.5)** обрезка слева/справа |
| `--crop-height` | `80` | высота ROI, если нет `--rising-y` |
| `--contrast` | `1.0` | OpenCV alpha |
| `--brightness` | `0.0` | OpenCV beta |
| `--gamma` | `1.0` | ≥ 0.05 |
| `--clahe` / `--no-clahe` | выкл. | CLAHE |
| `--clahe-clip` | `2.5` | сила CLAHE |
| `--clahe-grid` | `8` | тайлы (≥ 2) |
| `--blur` | `0` | `0/1` = выкл.; `≥2` Gaussian |
| `--invert` / `--equalize` | выкл. | доп. enhance |

---

## 2) Export — CINE → готовый датасет

Пишет enhanced-кропы в `data/dataset_<stem>/images/`.  
С `--predict` сразу гоняет YOLO и пишет `labels/`.

```powershell
python -m bubble_yolo export 24 `
  --from-preview data/preview_24/preview.json `
  --stride 500 `
  --predict `
  --save-overlay
```

Без preview (геометрия авто):

```powershell
python -m bubble_yolo export 24 --stride 500 --contrast 1.8 --brightness 30 --clahe --predict
```

### Флаги export

| Флаг | По умолчанию | Смысл |
|------|--------------|--------|
| `source` | — | stem / путь к `.cine` |
| `-o`, `--output` | `data/dataset_<stem>` | папка датасета |
| `--stride` | `500` | шаг по кадрам (≥ 1) |
| `--from-preview` | — | взять линии + enhance из `preview.json` |
| `--floor-y` / `--rising-y` | авто / из preview | геометрия ROI |
| `--side-crop-ratio` | `0.25` или из preview | **[0.0, 0.5)** |
| `--crop-height` | `80` | если нет `--rising-y` |
| enhance-флаги | как у preview | `--contrast`, `--clahe`, … |
| `--predict` | выкл. | сразу YOLO → `labels/` |
| `--weights` | лучший `yolo/runs/.../best.pt` | путь к `.pt` |
| `--conf` | `0.25` | порог детекции |
| `--imgsz` | `640` | размер входа YOLO |
| `--save-overlay` | выкл. | превью боксов |

Результат:

```text
data/dataset_24/
  images/
  labels/          # если --predict
  labels_overlay/  # если --save-overlay
  classes.txt
  export.json
```

Потом правь labels (Label Studio / руками) и сохраняй как `dataset_24_clean`.

---

## 3) Собрать train/val

Split **80/20 внутри каждого** `dataset_*_clean`:

```powershell
python -m bubble_yolo prepare --all-clean
```

Или: `python -m bubble_yolo prepare 12 13 x1 x2`

Результат: `yolo/dataset/` + `data.yaml`.

---

## 4) Обучить

```powershell
python -m bubble_yolo train --model yolo11s.pt
python -m bubble_yolo train --sweep
python -m bubble_yolo train --models yolov8n.pt yolov8s.pt yolo11n.pt yolo11s.pt
```

Веса: `yolo/runs/bubbles_*/weights/best.pt`.

---

## 5) Преразметить уже экспортированный датасет

Если export был без `--predict`:

```powershell
python -m bubble_yolo predict --source data/dataset_24/images --save-overlay --conf 0.25
```

---

## Краткий чеклист

1. Положить `.cine` в `data/films/`  
2. `preview` → проверить линии / enhance  
3. `export --from-preview … --predict` → `images/` + `labels/`  
4. Поправить разметку → `dataset_*_clean`  
5. `prepare --all-clean` → `train` → лучше модель  
6. Новые ролики снова через `export --predict`
