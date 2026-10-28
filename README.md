# Hybrid Time Series Forecasting

Самостоятельный проект для сравнения AR/ARIMA, пяти нейронных сетей,
трёх замороженных фундаментальных моделей и гибридов статистическая модель + FM.
Соседний репозиторий для установки, запуска и тестирования не требуется.

## Модели

Каждая модель находится в собственном файле `src/tshybrid/models/`:

| Модель | Файл | Реализация |
|---|---|---|
| AR | [ar.py](src/tshybrid/models/ar.py) | NumPy-регрессия |
| ARIMA | [arima.py](src/tshybrid/models/arima.py) | statsmodels |
| DLinear | [dlinear.py](src/tshybrid/models/dlinear.py) | Перенесённая нейронная сеть |
| TSMixer | [tsmixer.py](src/tshybrid/models/tsmixer.py) | Перенесённая нейронная сеть |
| ModernTCN | [moderntcn.py](src/tshybrid/models/moderntcn.py) | Перенесённая нейронная сеть |
| PatchTST | [patchtst.py](src/tshybrid/models/patchtst.py) | Перенесённая нейронная сеть |
| iTransformer | [itransformer.py](src/tshybrid/models/itransformer.py) | Перенесённая нейронная сеть |
| TimesFM | [timesfm.py](src/tshybrid/models/timesfm.py) | Официальный timesfm API |
| Chronos-Bolt | [chronos.py](src/tshybrid/models/chronos.py) | Официальный chronos-forecasting API |
| Moirai | [moirai.py](src/tshybrid/models/moirai.py) | Официальный Uni2TS API |

Пять нейронных сетей и необходимые слои включены в исходники проекта. Их источник
и лицензия указаны в [THIRD_PARTY.md](THIRD_PARTY.md); внешних импортов этих сетей нет.
Runner обучает Base-варианты. Классы Hyper сохранены в файлах сетей, но через
runner пока не выбираются. Гибриды реализованы в [hybrid.py](src/tshybrid/models/hybrid.py).

## Установка и данные

Проверен Python 3.12, Windows и CPU. Из каталога проекта, PowerShell:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[chronos,timesfm,plots,network,dev]"

uv venv .venv-moirai --python 3.12
uv pip install --python .venv-moirai/Scripts/python.exe -e ".[moirai]"
```

У Moirai отдельное окружение из-за ограничений версий зависимостей Uni2TS.
Для CUDA нужна соответствующая сборка PyTorch. Восстановление проверенных версий
описано в [RUN_GUIDE.md](RUN_GUIDE.md).

Поместите `ETTm1.npy`, `weather.npy`, `electricity.npy` в `data/`.
Формат NPY: числовая матрица `[time, channels]`. Можно скопировать их из любой
своей папки командой:

```powershell
.\.venv\Scripts\python.exe -m tshybrid prepare-data --source C:\datasets\mvts
.\.venv\Scripts\python.exe -m tshybrid inspect configs/benchmark.yaml
.\.venv\Scripts\python.exe -m tshybrid prefetch configs/benchmark.yaml
```

`prefetch` скачивает три checkpoints с ревизиями из `configs/model-lock.json`.
Поддерживается также wide CSV; официальные ETT можно скачать через `prepare-data`
без `--source`, затем указать CSV-пути в своей копии конфига.

## Полный запуск

`benchmark.yaml`: ETTm1/Weather/ECL, все каналы и окна, контекст 336,
горизонты 48/96, пять seed для каждой нейронной сети. Эпохи:
DLinear 50, TSMixer 25, ModernTCN 15, PatchTST 10, iTransformer 25.
Это общий протокол сравнения, без автоматического сравнения с авторскими логами.

Запускайте окружения последовательно:

```powershell
.\.venv\Scripts\python.exe -m tshybrid run configs/benchmark.yaml --models ar arima dlinear tsmixer moderntcn patchtst itransformer chronos timesfm arima_chronos arima_timesfm
.\.venv-moirai\Scripts\python.exe -m tshybrid run configs/benchmark.yaml --models moirai arima_moirai
```

Для `arima_moirai` автоматически запускается отдельная ARIMA-база в том же
окружении. `--datasets ETTm1` ограничивает датасет; `--models dlinear` — модель.
Полная серия дорогая, особенно ARIMA с подгонкой на каждом окне и канале.

Другие полные конфиги:

| Конфиг | Назначение |
|---|---|
| `hn_models.yaml` | Пять нейронных сетей, ETTm1, H=48/96, seed=0 |
| `hn_mvts_code.yaml` | Фиксированный профиль обучения DLinear, три датасета, четыре горизонта |
| `reproduce_ettm1_h48.yaml` | Тот же профиль DLinear только на ETTm1, H=48 |

Названия двух последних профилей отражают происхождение протокола; они читают
только данные и код этого проекта. Проверочные конфиги smoke/laptop/pilot удалены.

## Результаты и проверки

Каждый запуск печатает `outputs/<run-id>/report.html`. Там сохраняются метрики,
прогнозы Parquet, составляющие гибрида, лучшая validation-модель, история обучения,
выбранные параметры, ошибки, снимок кода и сведения об окружении.

```powershell
.\.venv\Scripts\python.exe -m tshybrid combine outputs/<core-id> outputs/<moirai-id> --output outputs/combined
Invoke-Item .\outputs\combined\report.html

.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests
```

Обычные отчёты и объединение совместимых запусков сохранены. Команды импорта
авторских логов, сравнения с эталонами и расширенного анализа гибридов удалены.
Исторические артефакты в `reports/` и `outputs/` остаются записями прежних запусков.

Подробности: [RUN_GUIDE.md](RUN_GUIDE.md), [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md),
[протокол](docs/PROTOCOL.md).
