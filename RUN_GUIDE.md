# Запуск time-series-hybrid

Проект устанавливается и тестируется самостоятельно, без соседних репозиториев.
Карта моделей находится в [README](README.md). Ниже команды PowerShell из каталога проекта.

## Окружения

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[chronos,timesfm,plots,network,dev]"
uv venv .venv-moirai --python 3.12
uv pip install --python .venv-moirai/Scripts/python.exe -e ".[moirai]"
```

Для восстановления сохранённых Windows/CPU-версий вместо установки extras:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-core.lock.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
uv pip install --python .venv-moirai/Scripts/python.exe -r requirements-moirai.lock.txt
uv pip install --python .venv-moirai/Scripts/python.exe --no-deps -e .
```

Для GPU установите сборку PyTorch с CUDA в нужное окружение. `device: auto` в YAML
выбирает `cuda:0` при доступности CUDA; допустимы также `cpu` и `cuda:0`.
CPU-потоки задаются параметром `threads`. Moirai использует собственное окружение.

## Данные

Полный benchmark ожидает `data/ETTm1.npy`, `data/weather.npy`, `data/electricity.npy`.
Их можно положить туда вручную или скопировать из своей папки:

```powershell
.\.venv\Scripts\python.exe -m tshybrid prepare-data --source C:\datasets\mvts
```

NPY должен содержать матрицу `[time, channels]`. При копировании сохраняются
исходный путь и SHA256; другой существующий файл не перезаписывается.
Никакой фиксированной структуры внешнего репозитория не требуется.

Для загрузки официальных ETT:

```powershell
.\.venv\Scripts\python.exe -m tshybrid prepare-data --datasets ETTm1 ETTm2 --destination data
```

Команда создаёт CSV. В копии конфига замените пути на `../data/ETTm1.csv` и,
если нужен ETTm2, добавьте его спецификацию. Weather и Electricity автоматически
не скачиваются. Wide CSV может содержать колонку `date`; timestamps должны быть
равномерными, уникальными и возрастающими. NaN/Inf отклоняются.

```powershell
.\.venv\Scripts\python.exe -m tshybrid inspect configs/benchmark.yaml
```

## Веса фундаментальных моделей

В `configs/model-lock.json` закреплены ревизии Chronos-Bolt tiny,
TimesFM 2.5 200M и Moirai 1.1 small. Кэш: `.cache/huggingface/`.

```powershell
.\.venv\Scripts\python.exe -m tshybrid prefetch configs/benchmark.yaml
```

При необходимости системного хранилища сертификатов:

```powershell
.\.venv\Scripts\python.exe -m tshybrid --system-certs prefetch configs/benchmark.yaml
```

Ревизии обновляются только явной командой `lock-models configs/benchmark.yaml`.

## Запуск

```powershell
.\.venv\Scripts\python.exe -m tshybrid run configs/benchmark.yaml --models ar arima dlinear tsmixer moderntcn patchtst itransformer chronos timesfm arima_chronos arima_timesfm
.\.venv-moirai\Scripts\python.exe -m tshybrid run configs/benchmark.yaml --models moirai arima_moirai
```

Зависимости выбранных гибридов подключаются автоматически. У Moirai-гибрида
собственная ARIMA-база `arima_moirai_base` в том же окружении, чтобы можно было
объединить результаты без дублирования ID основной ARIMA.

Отдельная модель и датасет:

```powershell
.\.venv\Scripts\python.exe -m tshybrid run configs/benchmark.yaml --models dlinear --datasets ETTm1
```

Сохранены полные конфиги `hn_models.yaml`, `hn_mvts_code.yaml` и
`reproduce_ettm1_h48.yaml`. Они используют только собственные файлы проекта.
Пути разрешаются относительно каталога конечного YAML. `extends` объединяет
словари, списки заменяются целиком. Для своих экспериментов можно создавать
копии конфигов; готовые уровни smoke/laptop/pilot удалены.

## Артефакты

В начале печатается путь `outputs/<run-id>`. В нём:

- `report.html`, `metrics.csv`, `aggregate.csv` — метрики и агрегация по seed;
- `manifest.json`, `resolved.yaml`, `source.zip` — конфиг, код и окружение;
- `selection.json` — оценки validation-кандидатов;
- `failures.json` — ошибки; полностью успешный запуск содержит `[]`;
- `<task>/predictions.parquet` — targets, прогнозы, ошибки и составляющие гибрида;
- `<task>/origins.npz`, `by_lead.csv` — точные окна и метрики по шагам;
- `<task>/best.pt`, `training-*.csv` — лучшая validation-модель и история обучения.

Неполный запуск возвращает exit code 1. Наличие HTML само по себе не означает успех.

```powershell
.\.venv\Scripts\python.exe -m tshybrid report outputs/<run-id>
.\.venv\Scripts\python.exe -m tshybrid combine outputs/<core-id> outputs/<moirai-id> --output outputs/combined
```

Для `combine` нужен новый каталог. Проверяются данные, каналы, splits, сезонность,
точные origins и отсутствие дублей model/seed. `report` пересоздаёт обычный HTML
и производные таблицы в каталоге запуска. Расширенный исследовательский отчёт
гибридов и сравнение с внешними эталонами удалены.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests
```

Тесты используют локальные модели и синтетические данные; исходники преподавателя
и сторонние checkpoints для них не нужны. Проверяются обучение всех пяти сетей,
checkpoint, причинность остатков, составляющие гибрида, метрики, CLI и совместимость
объединяемых запусков. Исторические отчёты не являются инструкциями для текущего CLI.
