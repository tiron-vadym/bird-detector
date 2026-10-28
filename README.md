# Bird detector — YOLO26

Проєкт донавчає `YOLO26n` для одного класу — `bird` — і запускає інференс на
відео, зберігаючи кожен кадр із bounding box, назвою класу та confidence.

## Встановлення

Потрібен Python 3.10+ та, для навчання на GPU, сумісні NVIDIA driver/CUDA.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Дані

Використано відкритий [Bird Detection YOLO Dataset](https://github.com/zhizhen777/bird-detection-yolo-dataset)
(CC BY 4.0): один клас `bird`, готові YOLO-розмітки та авторський train/val
split. Щоб зберегти навчання доступним на CPU, проєкт завантажує детерміновану
малу підвибірку: 100 train і 20 validation зображень. На кожне зображення є
`.txt` із рядками `class x_center y_center width height`, де координати
нормалізовано до 0–1.

```bash
python prepare_dataset.py
```

Ця команда створює `data/birds/images/{train,val}` та відповідні
`data/birds/labels/{train,val}`. Конфігурація — [data/bird.yaml](data/bird.yaml).

## Навчання

```bash
python prepare_dataset.py
python train.py
```

Типовий запуск: 50 епох, розмір 640, batch 8. За нестачі VRAM зменште batch
або використайте автоматичний: `python train.py --batch -1`. Після успішного
навчання найкращий чекпойнт копіюється у `weights/best.pt`; усі графіки,
метрики та проміжні ваги залишаються в `runs/bird_yolo26n/`.

## Відеоінференс

```bash
python predict_video.py --input examples/input.mp4 --output examples/output.mp4
```

Додатково можна передати `--weights path/to/model.pt`, `--conf 0.35`,
`--device 0` або `--device cpu`.

`examples/input.mp4` — короткий 20-кадровий MP4, зібраний із validation
зображень, а `examples/output.mp4` — фактичний результат запуску команди вище
з `weights/best.pt`.

## Метрики validation

Перевірка `weights/best.pt` на 20 validation зображеннях (YOLO26n, 320 px,
CPU) дала такі результати. Це чекпойнт після першої завершеної епохи короткого
CPU baseline-запуску:

| Precision | Recall | mAP@50 | mAP@50–95 |
| ---: | ---: | ---: | ---: |
| 0.826 | 0.700 | 0.847 | 0.563 |

Метрики й візуалізації validation-запуску збережено в
`runs/detect/runs/validation_final/`. Це короткий baseline на малій вибірці,
а не оцінка узагальнення на незалежному польовому відео.

## Відомі обмеження та покращення

- Невеликий датасет гірше узагальнюється на дрібних, розмитих або частково
  закритих птахах і на відео з іншою камерою.
- Доцільно додати власні відеокадри, особливо складні фони та негативні кадри,
  і перевірити, що train/val не містять майже ідентичних сусідніх кадрів.
- Для вищої якості спробуйте більше епох, підбір `imgsz`/аугментацій і більшу
  модель після фіксації базової метрики; для стабільності відео можна додати
  трекінг (ByteTrack) або згладження детекцій.

## Очікувана структура після запуску

```text
bird-detector/
├── data/bird.yaml
├── prepare_dataset.py
├── examples/input.mp4
├── examples/output.mp4
├── weights/best.pt
├── train.py
└── predict_video.py
```
