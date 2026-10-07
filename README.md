# Bubble Dataset

Двухшаговый пайплайн преразметки пузырей из Phantom `.cine` под YOLO + ручная доработка в Label Studio.

**Классы:** `0 = rising`, `1 = sitting`  
Пустой кадр → пустой `.txt` (отдельный класс «пустота» для detect не нужен).

---

## Установка пайплайна

```powershell
cd ...\Bubbles
pip install -e .
```

---

## 1) Preview — линии, enhance и палитра алгоритмов поиска

### Куда класть `.cine`

Положи файлы в:

```text
data/films/
```

Пример: `data/films/24.cine`.

Дальше можно вызывать по номеру, имени или пути:


| Аргумент `source`     | Что подхватится      |
| --------------------- | -------------------- |
| `24`                  | `data/films/24.cine` |
| `24.cine`             | `data/films/24.cine` |
| `data/films/24.cine`  | этот путь            |
| полный путь к `.cine` | как указано          |


Поиск идёт через `resolve_source`: сначала путь как есть, потом `data/films/<имя>`, затем `data/films/<stem>.cine`.

Работай из корня проекта (`...\Bubbles`), иначе относительный `data/films` не найдётся.

### Запуск

```powershell
python -m bubble_dataset preview 24 --contrast 1.8 --brightness 30 --clahe
python -m bubble_dataset list-presets
```



### Флаги preview



#### Источник и вывод


| Флаг                   | По умолчанию          | Границы / смысл                                 |
| ---------------------- | --------------------- | ----------------------------------------------- |
| `source` (позиционный) | —                     | stem / имя / путь к `.cine` или папке с кадрами |
| `-o`, `--output`       | `data/preview_<stem>` | папка результата                                |




#### Геометрия кадра (линии и ROI)


| Флаг                | По умолчанию            | Границы / смысл                                         |
| ------------------- | ----------------------- | ------------------------------------------------------- |
| `--frame-index`     | середина ролика (авто)  | целое ≥ 0, индекс кадра в `.cine`                       |
| `--floor-y`         | авто (`detect_floor_y`) | целое, строка дна; должно быть **>** `rising_y`         |
| `--rising-y`        | `floor_y − crop_height` | целое ≥ 0; должно быть **<** `floor_y`                  |
| `--side-crop-ratio` | `0.25`                  | **[0.0, 0.5)** — доля ширины, обрезаемая слева и справа |
| `--crop-height`     | `80`                    | целое ≥ 1; высота ROI, если `--rising-y` не задан       |


Если `rising_y >= floor_y`, preview упадёт с ошибкой.

#### Enhance (улучшение картинки)


| Флаг           | По умолчанию | Границы / смысл                                                                  |
| -------------- | ------------ | -------------------------------------------------------------------------------- |
| `--contrast`   | `1.0`        | float > 0 (OpenCV `alpha`); `1.0` = без изменений; типично `0.5…3.0`             |
| `--brightness` | `0.0`        | float (OpenCV `beta`); типично **−255…255**; `0` = без сдвига                    |
| `--gamma`      | `1.0`        | float; в коде режется снизу до **≥ 0.05**; `< 1` поднимает тени, `> 1` затемняет |
| `--clahe`      | выкл.        | включить CLAHE                                                                   |
| `--no-clahe`   | —            | явно выключить CLAHE (нужно при `--from-preview`, если в meta был включён)       |
| `--clahe-clip` | `2.5`        | float > 0; сила CLAHE (`clipLimit`)                                              |
| `--clahe-grid` | `8`          | целое; размер тайла; в коде не меньше **2**                                      |
| `--blur`       | `0`          | целое; `0` или `1` = выкл.; `≥ 2` — Gaussian blur (ядро делается нечётным)       |
| `--invert`     | выкл.        | инверсия яркости                                                                 |
| `--equalize`   | выкл.        | гистограммное выравнивание (`equalizeHist`)                                      |


Пример подбора:

```powershell
python -m bubble_dataset preview 24 `
  --contrast 1.8 `
  --brightness 30 `
  --gamma 0.8 `
  --clahe `
  --clahe-clip 2.5 `
  --clahe-grid 8 `
  --side-crop-ratio 0.25 `
  --crop-height 80
```



### Пресеты поиска (палитра)

**contours:** `01_otsu` `02_adaptive` `03_blackhat` `04_tophat` `05_dog` `06_laplacian` `07_sobel_mag`  
**blobs:** `08_blob` `09_hough` `10_msers`  
**motion:** `11_bg_absdiff` `12_frame_diff`

Список из CLI: `python -m bubble_dataset list-presets`.

### Что получится

В `data/preview_<stem>/` (или в `-o`):

- `original_guides.jpg` — кадр с линиями `floor_y` / `rising_y` и боковыми границами  
- `enhanced_crop.png` — ROI после enhance  
- `palette.jpg` — все пресеты поиска на одном листе  
- `cells/` — отдельные ячейки палитры  
- `preview.json` — линии, `image_params` и счётчики детекций (для `--from-preview` на шаге export)

---



## 2) Export — выбранный алгоритм поиска

```powershell
python -m bubble_dataset export 24 `
  --from-preview data/preview_24/preview.json `
  --preset 03_blackhat `
  --stride 500 `
  --debug
```

`--from-preview` подтягивает линии и `image_params`. Пресет можно указывать коротко: `--preset blackhat`.

Результат: `images/` (enhanced) + `labels/` (YOLO) + `debug/`.

---



## 3) Ручная доработка в Label Studio

Label Studio удобнее AnyLabeling для правки уже готовых YOLO-боксов.  
Ниже рабочий путь под Windows (проверен на этом проекте).

### 3.1. Установка

```powershell
pip install -U label-studio
```

Команда `label-studio` часто **не попадает в PATH**. Найди и запусти exe одной командой:

```powershell
& (Get-ChildItem -Recurse "$env:LOCALAPPDATA\Packages" -Filter "label-studio.exe" -ErrorAction SilentlyContinue |
  Select-Object -First 1 -ExpandProperty FullName)
```

После старта открой: [http://localhost:8080](http://localhost:8080)  
Терминал не закрывай — пока он открыт, сервер работает.

> Первый запуск может минуту молчать (миграции БД). Предупреждения про `.env` / `SECRET_KEY` нормальны.



### 3.2. Создать проект и настроить классы

1. **Create Project**
2. **Settings → Labeling Interface → Code** — вставь:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="rising" background="green"/>
    <Label value="sitting" background="orange"/>
  </RectangleLabels>
</View>
```

1. Save

Имена `rising` / `sitting` должны совпадать с `classes.txt`.

### 3.3. Подготовить JSON с картинками и боксами

Label Studio **не читает** YOLO `.txt` напрямую. Нужен JSON с картинками внутри (base64) — иначе на Windows часто падает загрузка по пути:

> There was an issue loading URL from $image value

Из корня проекта одной командой (имя датасета → папка `data/dataset_<имя>`):

```powershell
python -m bubble_dataset.to_label_studio 24
```

Скрипт читает `data/dataset_24/images/` + `labels/` (+ `classes.txt`) и пишет:

```text
data/dataset_24/ls_tasks_embedded.json
```

Аргумент можно передать так:


| Аргумент          | Папка             |
| ----------------- | ----------------- |
| `24`              | `data/dataset_24` |
| `dataset_24`      | `data/dataset_24` |
| `data/dataset_24` | как указано       |


Другой путь вывода: `-o path\to\out.json`.

### 3.4. Импорт в проект

1. В проекте: **Import**
2. Выбери файл:
  ```
   data/dataset_24/ls_tasks_embedded.json
  ```
3. Открой любую задачу — должны быть видны картинка и готовые прямоугольники.



### 3.5. Как править


| Действие              | Как                                        |
| --------------------- | ------------------------------------------ |
| Листать кадры         | стрелки / список tasks слева               |
| Выбрать класс         | `rising` или `sitting` на панели           |
| Новый бокс            | зажать ЛКМ и нарисовать                    |
| Подвинуть / ресайз    | тянуть края существующего бокса            |
| Сменить класс у бокса | выделить → выбрать другой Label            |
| Удалить бокс          | выделить → Delete                          |
| Сохранить кадр        | **Submit** (новый) или **Update** (правка) |


На что смотреть:

- ложный шум / блики — удалить;
- слипшиеся пузыри — разделить на 2 бокса;
- `sitting` только у пузырей на дне;
- пустой кадр — оставь без боксов и Submit.



### 3.6. Экспорт обратно в YOLO

**Не используй** `YOLO / YOLO with Images` после импорта через `ls_tasks_embedded.json` (base64).  
Label Studio не знает имён файлов: папка `images/` будет пустой, а `labels/*.txt` получат мусорные имена.

Рабочий путь:

1. В Label Studio: **Export** → формат **JSON**
2. Сохрани файл, например `Downloads\project-24.json`
3. Из корня проекта:

```powershell
python -m bubble_dataset.from_label_studio 24 "...\Downloads\project-24.json"
```

Скрипт:

- читает правки из JSON;
- копирует картинки из `data/dataset_24/images/`;
- пишет YOLO-labels с нормальными именами (`frame_….txt`);
- результат: `data/dataset_24_clean/` (`images/` + `labels/` + `classes.txt`).

Другой выход: `-o data/dataset_24_clean`.

### 3.7. Типичные проблемы


| Симптом                           | Причина                      | Решение                                        |
| --------------------------------- | ---------------------------- | ---------------------------------------------- |
| пустой `images/`                  | Импорт был через base64      | Export **JSON** + `from_label_studio` (§3.6)   |
| `labels` с именами `...CYII=.txt` | То же                        | То же                                          |
| Классы не те / боксы пустые       | Не совпал Labeling Interface | Имена `rising`/`sitting`, теги `image`/`label` |
| Дубликаты tasks                   | Два импорта подряд           | Удали старые tasks перед новым Import          |




## Краткий чеклист

1. `preview` (+ enhance) → смотри `palette.jpg`, выбери пресет поиска
2. `export --preset … --from-preview …` → `images/` + `labels/`
3. Label Studio → ручная правка
4. Export **JSON** → `from_label_studio` → `dataset_*_clean/`

