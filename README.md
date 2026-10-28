# Hybrid Time Series Forecasting

## Полная документация

- [RUN_GUIDE.md](RUN_GUIDE.md) — установка, данные, веса, все режимы запуска, отчёты и устранение ошибок.
- [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md) — устройство проекта, разбор фрагментов кода и карта реализации.
- [SUPERVISOR_TASK_REVIEW.md](SUPERVISOR_TASK_REVIEW.md) — выполнение задачи руководителя, реальные результаты и недостающие проверки.

Актуальный исправленный контроль DLinear: [reproduction_corrected](reports/reproduction_corrected/report.html).
В раннем HN-профиле было лишнее усечение окон; оно исправлено, общий режим гибридов не затронут.

Контур для сравнения статистических моделей, DLinear, замороженных FM и коррекции остатков.
Реализованы LastValue, SeasonalNaive, AR, ARIMA, ETS, Linear,
локальные реализации DLinear, TSMixer, ModernTCN, PatchTST, iTransformer,
Chronos-Bolt, TimesFM 2.5, Moirai 1.1 и гибрид статистическая модель + FM.
Есть общий rolling-origin evaluator, выбор настроек по validation,
сохранение прогнозов в Parquet, сравнение с эталонами и HTML-отчёты.

## Нейронные модели на ноутбуке

Пять моделей реализованы в [models/networks.py](src/tshybrid/models/networks.py)
и [models/moderntcn.py](src/tshybrid/models/moderntcn.py), общие слои —
[models/layers.py](src/tshybrid/models/layers.py). Архитектуры соответствуют
Base-вариантам HN-MVTS; импортов его реализаций в рабочем коде нет.
Происхождение архитектур и адаптированного кода указано в [THIRD_PARTY.md](THIRD_PARTY.md).

Из каталога `time-series-hybrid`, PowerShell:

```powershell
# Установка без фундаментальных моделей и загрузки их весов.
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[dev]" --constraint requirements-core.lock.txt
.\.venv\Scripts\python.exe -m tshybrid prepare-data --source ../HN-MVTS/data

# Все пять локальных сетей: проверка, затем небольшой CPU-пилот.
.\.venv\Scripts\python.exe -m tshybrid run configs/hn_models_smoke.yaml
.\.venv\Scripts\python.exe -m tshybrid run configs/hn_models_laptop.yaml
```

Пилот сохраняет все 7 каналов ETTm1 и исходные размеры сетей, но использует
512 train-окон, 2 эпохи и по 16 validation/test origins, H=48. Его метрики
не являются воспроизведением статьи. Полный конфиг без сокращения окон —
`configs/hn_models.yaml` (ETTm1, H=48/96, seed=0); он существенно дороже.

**Начать с [результатов первых экспериментов](reports/README.md).**
Подробности: [протокол](docs/PROTOCOL.md), [воспроизведение статей](docs/REPRODUCTION.md).

Пункты 5 и 6 выделены в отдельное исследование: [гибриды и первый отчёт](reports/hybrid_study/REPORT.md).
Конфиги `hybrid_core.yaml` и `hybrid_moirai.yaml` сравнивают ARIMA / FM / ARIMA+FM
для всех трёх FM: ETTm1 (7 каналов), Weather (21), ECL (16 фиксированных каналов),
16 validation/test origins, H=48/96. Это расширенный pilot, не полная оценка всех окон.

```powershell
# Одной командой: оба окружения, затем отчёт (каталог отчёта должен быть новым).
.\.venv\Scripts\python.exe scripts/run_hybrid_study.py --output reports/<new-study>

# Запускать последовательно, чтобы не конкурировать за CPU.
.\.venv\Scripts\python.exe -m tshybrid run configs/hybrid_core.yaml
.\.venv-moirai\Scripts\python.exe -m tshybrid run configs/hybrid_moirai.yaml
.\.venv\Scripts\python.exe -m tshybrid hybrid-report outputs/<core-id> outputs/<moirai-id> --output reports/<new-study> --reproduction reports/reproduction
```

`hybrid-report` проверяет одинаковые targets и сохранённые составляющие гибрида,
затем создаёт `REPORT.md`, `report.html`, таблицы выигрыша, разбор по каналам/origins/lead,
графики и копии provenance. В прогнозах сохраняются отдельно база и коррекция.
У Moirai собственный baseline `arima_moirai_base` в том же окружении, чтобы не
приписать гибриду различия численных библиотек.
В этих двух конфигах явно задана `initialization: approximate_diffuse`: на расширенном
Weather стационарная инициализация исходного ARIMA приводила к LU decomposition error.
Численно неудачный validation-кандидат записывается в `selection.json`; выбор продолжается
среди остальных. Ошибка выбранной модели на test по-прежнему делает прогон неполным.

## Запуск в текущем workspace

Окружения, данные и три checkpoints уже подготовлены. PowerShell:

```powershell
cd C:\workspace\learning\research\time-series-hybrid

# Проверка данных, без загрузки моделей.
.\.venv\Scripts\python.exe -m tshybrid inspect configs/pilot.yaml

# Короткий полный путь: статистика, DLinear, FM, гибрид.
.\.venv\Scripts\python.exe -m tshybrid run configs/smoke.yaml

# ETTm1/Weather/ECL, H=48/96, по 2 канала и 8 origins.
.\.venv\Scripts\python.exe -m tshybrid run configs/pilot.yaml

# Остальные FM на тех же данных и окнах.
.\.venv\Scripts\python.exe -m tshybrid run configs/foundation.yaml --models timesfm
.\.venv-moirai\Scripts\python.exe -m tshybrid run configs/foundation.yaml --models moirai

# Полный ETTm1, H=48, 50 эпох, seed=0, авторский протокол.
.\.venv\Scripts\python.exe -m tshybrid run configs/reproduce_ettm1_h48.yaml
```

Каждая команда печатает путь `outputs/<run-id>/report.html`.
Открыть: `Invoke-Item .\outputs\<run-id>\report.html`.
Пилот проверяет весь путь, но не считается воспроизведением статьи.
Локально проверен CPU; поддержка CUDA предусмотрена, но здесь не проверена.

## Установка с нуля

Python 3.11+; проверен 3.12. Moirai использует отдельное окружение из-за ограничений
Uni2TS на PyTorch/NumPy. Команды с `uv`:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[chronos,timesfm,plots,network,dev]"
uv venv .venv-moirai --python 3.12
uv pip install --python .venv-moirai/Scripts/python.exe -e ".[moirai]"

.\.venv\Scripts\python.exe -m tshybrid prepare-data --source ../HN-MVTS/data/datasets
.\.venv\Scripts\python.exe -m tshybrid import-hn-reference --repo ../HN-MVTS
.\.venv\Scripts\python.exe -m tshybrid prefetch configs/foundation.yaml
```

Без uv: `python -m venv .venv`, затем `.venv\Scripts\python -m pip install -e ".[chronos,timesfm,plots,network,dev]"`.
Точные версии текущих Windows/CPU-окружений: `requirements-core.lock.txt` и
`requirements-moirai.lock.txt`. Для восстановления установить соответствующий
lock-файл, затем `uv pip install --no-deps -e .` в этом же окружении.
GPU-версию PyTorch подбирать под CUDA отдельно.

Для сети с корпоративным сертификатом доступна опция `--system-certs`:
`python -m tshybrid --system-certs prefetch configs/foundation.yaml`.
Она использует системное хранилище через truststore, сохраняя проверку TLS.

Без соседнего HN-MVTS доступна загрузка официального ETT:
`python -m tshybrid prepare-data --datasets ETTm1 ETTm2 --destination data`.
Она создаёт CSV; в копии конфига нужно заменить `../data/ETTm1.npy` на
`../data/ETTm1.csv`. SHA авторского NPY-эталона при этом не совпадёт.
Weather/Electricity автоматически из сторонних зеркал не скачиваются.

## Конфиги

| Конфиг | Назначение |
|---|---|
| `smoke.yaml` | ETTm1, H=48, 2 канала, 2 origins, DLinear 2 эпохи |
| `pilot.yaml` | Три датасета, H=48/96, 2 канала, 8 origins, DLinear 5 эпох |
| `foundation.yaml` | Три FM на том же пилоте, запуск по окружениям |
| `benchmark.yaml` | Все каналы/origins, H=48/96, DLinear 50 эпох и 5 seed |
| `hn_mvts_code.yaml` | Авторский протокол, H=48/96/192/336, три датасета |
| `reproduce_ettm1_h48.yaml` | Один полный воспроизводящий прогон |

Пути разрешаются относительно каталога конфига, в том числе при `extends`.
Словари объединяются, списки заменяются. `device: auto` выбирает CUDA при наличии;
также допустимы `cpu` и `cuda:0`. CPU-потоки: `threads`.
`--models`/`--datasets` ограничивают запуск; зависимости гибрида подключаются автоматически.

`configs/model-lock.json` закрепляет SHA checkpoints. Обновление — отдельное действие:
`python -m tshybrid lock-models configs/foundation.yaml`.

Полный benchmark существенно дороже пилота, особенно ARIMA на каждом origin:

```powershell
.\.venv\Scripts\python.exe -m tshybrid run configs/benchmark.yaml --models last_value seasonal_naive ar arima ets dlinear chronos timesfm arima_chronos
.\.venv-moirai\Scripts\python.exe -m tshybrid run configs/benchmark.yaml --models moirai
```

Для абляции контекста создать копии с разными `context` и проверить `fit_prefix < context`.

## Артефакты

В `outputs/<run-id>/` сохраняются:

- `report.html`, `metrics.csv`, `aggregate.csv` — test-метрики и sample std по seed;
- `manifest.json`, `resolved.yaml` — конфиг, SHA данных/кода, splits/каналы,
  пакеты, устройство, checkpoints и выбранные параметры;
- `selection.json` — все кандидаты и их validation-оценки;
- `failures.json` — явные ошибки; неполный прогон возвращает exit code 1;
- `<dataset>-<model>-H<h>-s<seed>/predictions.parquet` — origin/channel/lead,
  truth, forecast, ошибки и квантили;
- `by_lead.csv`, `origins.npz` — метрики по шагам и точные индексы;
- `best.pt`, `training-*.csv` — лучшая validation-модель DLinear и история обучения;
- `mse-*.png` — standalone графики при установленном extra `plots`.

Перестроить HTML: `python -m tshybrid report outputs/<run-id>`.
Объединить отдельные окружения:

```powershell
.\.venv\Scripts\python.exe -m tshybrid combine outputs/<pilot-id> outputs/<timesfm-id> outputs/<moirai-id> --output outputs/combined
```

Проверяются SHA данных, каналы, splits, сезонность и точные origins.
Несовместимые протоколы и дубли model/seed отклоняются.

## Проверки и дальнейшая работа

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests
```

Проверяются границы окон, train-only scaling, причинность остатков AR/ARIMA,
отсутствие влияния test на обучение DLinear, последний батч, интервальные метрики,
Parquet и защита от сравнения pilot с опубликованным эталоном.
При наличии HN-MVTS проверяется forward DLinear против исходной реализации.

Полные матрицы seed/горизонтов, оригинальные FM-бенчмарки, проверка pretraining overlap,
M4/Monash, PatchTST/N-BEATS, критерий выбора коррекции и оценка устойчивости
остаются следующими исследовательскими этапами.
