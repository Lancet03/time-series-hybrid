# Полная инструкция запуска time-series-hybrid

Состояние проекта: **26 октября 2026 года**. Проверенный путь — Windows, PowerShell,
Python 3.12 и CPU. Ниже собраны установка, подготовка данных, все виды запусков,
повторное построение отчётов и диагностика ошибок. Для выполнения инструкции не
требуется читать остальные Markdown-файлы.

Описание реализации: [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md).
Сопоставление с задачей руководителя: [SUPERVISOR_TASK_REVIEW.md](SUPERVISOR_TASK_REVIEW.md).

## 1. Что запускается

Проект умеет сравнивать статистические модели, Linear/DLinear, три замороженные
фундаментальные модели и гибриды ARIMA+FM на остатках. Модели оцениваются по общим
историческим окнам; параметры выбираются по validation, итоговые метрики — по test.

В текущем workspace уже подготовлены:

- `.venv/` — статистика, DLinear, Chronos, TimesFM, графики и тесты;
- `.venv-moirai/` — отдельное окружение для Moirai;
- `data/ETTm1.npy`, `data/weather.npy`, `data/electricity.npy`;
- `.cache/huggingface/` — зафиксированные checkpoints трёх FM;
- отчёты первоначального и расширенного пилотов в `reports/`.

В git данные, окружения, cache и `outputs/` не включаются. После переноса только
репозитория их потребуется подготовить заново. Готовые компактные отчёты в `reports/`
можно читать без Python и без скачивания весов.

## 2. Быстрый путь в уже подготовленном workspace

Открыть PowerShell и перейти в проект:

```powershell
cd C:\workspace\learning\research\time-series-hybrid
```

Посмотреть имеющиеся результаты:

```powershell
Invoke-Item .\reports\hybrid_study\report.html
```

Проверить окружение и конфиг:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m tshybrid --help
.\.venv\Scripts\python.exe -m tshybrid inspect configs/smoke.yaml
```

Запустить небольшую проверку всего пути:

```powershell
.\.venv\Scripts\python.exe -m tshybrid run configs/smoke.yaml
```

Она использует ETTm1, два канала, H=48, по два validation/test origins и две эпохи
DLinear. Помимо статистики и DLinear запускаются настоящая Chronos-Bolt и гибрид.
Команда напечатает `Run directory: ...` и в конце `Report: .../report.html`.

Прогон успешен, если `manifest.json` содержит `status: completed`, а `failures.json`
равен `[]`. Наличие одного HTML-файла само по себе ещё не доказывает успех всех моделей.

## 3. Установка с нуля

### 3.1. Предварительные условия

Нужны исходники этого проекта, Python и доступ к интернету для установки пакетов
и первого скачивания весов. Проверена версия Python 3.12; `pyproject.toml` допускает
Python 3.11+. Остальные ОС и сборки Python отдельно не проверены.

Команды ниже используют `uv`. При его отсутствии можно использовать стандартный
`venv` и `pip`, как показано ниже. Активация окружения не обязательна: везде явно
указан нужный `python.exe`, что также обходит ограничения PowerShell на Activate.ps1.

Два окружения нужны из-за зависимостей Uni2TS: у Moirai проверены NumPy 1.26.4,
PyTorch 2.4.1 и PyArrow 19.0.1; основное окружение использует другой набор версий.
Не следует устанавливать все extras в одно окружение в расчёте на уже проверенную совместимость.

### 3.2. Восстановление проверенных версий

Если `.venv` и `.venv-moirai` уже готовы, этот шаг можно пропустить.
Из корня проекта:

```powershell
uv venv .venv --python 3.12 --cache-dir .cache/uv
uv pip install --python .venv/Scripts/python.exe --cache-dir .cache/uv -r requirements-core.lock.txt
uv pip install --python .venv/Scripts/python.exe --cache-dir .cache/uv --no-deps -e .

uv venv .venv-moirai --python 3.12 --cache-dir .cache/uv
uv pip install --python .venv-moirai/Scripts/python.exe --cache-dir .cache/uv -r requirements-moirai.lock.txt
uv pip install --python .venv-moirai/Scripts/python.exe --cache-dir .cache/uv --no-deps -e .
```

Lock-файлы фиксируют версии зависимостей проверенных Windows/CPU-окружений, но не
содержат сам локальный проект: поэтому последняя команда `-e .` необходима.
Это snapshots пакетов, а не универсальная гарантия идентичности всех бинарных сборок на любой ОС.

### 3.3. Установка по диапазонам pyproject.toml

Альтернатива lock-файлам, например для дальнейшей разработки:

```powershell
uv venv .venv --python 3.12 --cache-dir .cache/uv
uv pip install --python .venv/Scripts/python.exe --cache-dir .cache/uv -e ".[chronos,timesfm,plots,network,dev]"

uv venv .venv-moirai --python 3.12 --cache-dir .cache/uv
uv pip install --python .venv-moirai/Scripts/python.exe --cache-dir .cache/uv -e ".[moirai]"
```

В этом варианте разрешённые версии могут отличаться от выполненных экспериментов.
Фактические пакеты нового запуска сохранятся в его `manifest.json`.

Без uv, при доступной команде Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .

python -m venv .venv-moirai
.\.venv-moirai\Scripts\python.exe -m pip install -r requirements-moirai.lock.txt
.\.venv-moirai\Scripts\python.exe -m pip install --no-deps -e .
```

Если `python` указывает на неработающий Windows Store alias, использовать путь к
установленному Python либо `uv venv --python 3.12`. Путь к Python из чужого профиля
не следует переносить на другой компьютер.

### 3.4. Проверка CPU/CUDA

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
.\.venv-moirai\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

`device: auto` в YAML выберет CUDA при её доступности, иначе CPU. Для принудительного
CPU использовать `device: cpu` в копии конфига. `device: cuda:0` требует совместимой
GPU-сборки PyTorch и драйвера; CPU lock-файл сам по себе её не обеспечивает.
GPU-прогоны в этом проекте ещё не проверены, GPU VRAM не измеряется.

## 4. Подготовка данных

### 4.1. Из соседнего HN-MVTS

Ожидаемая структура:

```text
research/
  HN-MVTS/data/datasets/
    ETTm1.npy
    weather.npy
    electricity.npy
  time-series-hybrid/
```

Команда копирования:

```powershell
.\.venv\Scripts\python.exe -m tshybrid prepare-data --source ../HN-MVTS/data/datasets
```

Она копирует три файла в `data/` и записывает `data/manifest.json` с SHA-256.
Если целевой файл уже существует и отличается, команда откажется его перезаписывать.
Исходные данные HN-MVTS не изменяются. При необходимости ETTm2:

```powershell
.\.venv\Scripts\python.exe -m tshybrid prepare-data --source ../HN-MVTS/data/datasets --datasets ETTm2
```

Текущие конфиги используют:

| Датасет | Файл | Используемые точки | Полное число каналов | Сезонный период |
|---|---|---:|---:|---:|
| ETTm1 | `ETTm1.npy` | первые 57 600 из 69 680 | 7 | 96 |
| Weather | `weather.npy` | 52 696 | 21 | 144 |
| Electricity / ECL | `electricity.npy` | 26 304 | 321 | 24 |

Сезонный период задан в шагах и используется в SeasonalNaive/MASE. Частоты в
конфигах — 15 минут, 10 минут и 1 час соответственно. У NPY нет timestamps;
их корректность и порядок каналов наследуются от подготовки исходных файлов.

### 4.2. Официальный ETT CSV без соседнего репозитория

```powershell
.\.venv\Scripts\python.exe -m tshybrid prepare-data --datasets ETTm1 ETTm2 --destination data
```

Без `--source` автоматически скачиваются только официальные ETT CSV. Для запуска
нужно сделать копию YAML в том же `configs/`, оставить доступный датасет и заменить
`path: ../data/ETTm1.npy` на `path: ../data/ETTm1.csv`.
При отсутствии Weather/ECL не запускать конфиг, который всё ещё ссылается на них.
SHA CSV отличается от SHA авторского NPY, поэтому строгая проверка NPY-эталона
пометит такой эксперимент как несопоставимый.

### 4.3. Собственный набор

Поддерживаются двумерный NPY `[time, channels]` или wide CSV с числовыми каналами.
CSV может содержать колонку `date` либо имя из `date_column`. При наличии дат они
проверяются на порядок, повторы и регулярность. NaN/Inf отклоняются. Автоматической
импутации, ресемплинга и загрузки ragged-наборов M4/Monash пока нет.

## 5. Веса фундаментальных моделей

Используются следующие зафиксированные checkpoints:

| Модель | Checkpoint |
|---|---|
| Chronos-Bolt Tiny | `amazon/chronos-bolt-tiny` |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` |
| Moirai 1.1 Small | `Salesforce/moirai-1.1-R-small` |

Точные commit SHA находятся в `configs/model-lock.json`. Заранее скачать их:

```powershell
.\.venv\Scripts\python.exe -m tshybrid prefetch configs/foundation.yaml
```

Проверка cache сначала локальная; при отсутствии нужного snapshot выполняется
скачивание. После подготовки весов, данных и пакетов эксперименты не требуют
сетевого вызова для каждого прогноза.

Для доверенного корпоративного сертификата Windows:

```powershell
.\.venv\Scripts\python.exe -m tshybrid --system-certs prefetch configs/foundation.yaml
```

Нужен extra `network` или установленный `truststore`. Опция включает системное
хранилище сертификатов и сохраняет проверку TLS.

Обновление SHA — отдельное действие, меняющее постановку воспроизведения:

```powershell
.\.venv\Scripts\python.exe -m tshybrid lock-models configs/foundation.yaml --output configs/model-lock-next.json
```

Чтобы использовать новый файл, явно изменить `model_lock` в копии YAML. Для повторения
имеющихся результатов оставлять текущий `model-lock.json`.

## 6. Какой конфиг выбрать

| Конфиг | Содержание |
|---|---|
| `smoke.yaml` | ETTm1, 2 канала, H=48, 2 origins, DLinear 2 эпохи |
| `pilot.yaml` | 3 набора, по 2 канала, H=48/96, 8 origins, DLinear 5 эпох |
| `foundation.yaml` | Три FM на той же сетке первоначального пилота |
| `hybrid_core.yaml` | ARIMA, Chronos, TimesFM и два гибрида; 7/21/16 каналов, 16 origins |
| `hybrid_moirai.yaml` | Moirai, её ARIMA baseline и гибрид на той же расширенной сетке |
| `reproduce_ettm1_h48.yaml` | Полный ETTm1, DLinear H=48, 50 эпох, seed=0, профиль HN |
| `hn_mvts_code.yaml` | DLinear на трёх полных наборах, H=48/96/192/336, seed=0 |
| `benchmark.yaml` | Общая полная сетка H=48/96, все каналы; DLinear пять seed |

В двух гибридных конфигах ETTm1 и Weather имеют все каналы, ECL — фиксированные
16 индексов. Origins остаются подвыборкой: это расширенный pilot.

## 7. Повторение экспериментов

### 7.1. Первоначальный пилот

```powershell
.\.venv\Scripts\python.exe -m tshybrid run configs/pilot.yaml
.\.venv\Scripts\python.exe -m tshybrid run configs/foundation.yaml --models timesfm
.\.venv-moirai\Scripts\python.exe -m tshybrid run configs/foundation.yaml --models moirai
```

Фильтры запуска:

```powershell
.\.venv\Scripts\python.exe -m tshybrid run configs/pilot.yaml --models ar arima dlinear --datasets ETTm1
```

Для гибрида `--models arima_chronos` автоматически включает его базу и FM.
Названия в фильтре — поле `id` модели или `name` датасета, с учётом регистра.
Флага `--horizons` в CLI нет: горизонты меняются в копии конфига.

### 7.2. Пункты 5 и 6 одной командой

Создать новое имя итогового отчёта, чтобы сохранить прежний:

```powershell
$studyStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$studyOutput = "reports/hybrid-repeat-$studyStamp"
.\.venv\Scripts\python.exe scripts/run_hybrid_study.py --output $studyOutput
Invoke-Item (Join-Path $studyOutput 'report.html')
```

Скрипт последовательно выполняет `hybrid_core.yaml`, `hybrid_moirai.yaml` и
`hybrid-report`. Это 48 задач model/dataset/horizon и 18 сравнений тройками.
Полные логи находятся в `outputs/study-logs-<timestamp>/`.

Запуски идут последовательно, чтобы снизить конкуренцию за CPU. Стоимость такого
эксперимента значительно выше smoke. Точное время зависит от оборудования.

### 7.3. Те же шаги вручную

```powershell
.\.venv\Scripts\python.exe -m tshybrid run configs/hybrid_core.yaml
.\.venv-moirai\Scripts\python.exe -m tshybrid run configs/hybrid_moirai.yaml
```

Запомнить две напечатанные директории, затем построить отчёт. Вместо шаблонов
`<core-id>`, `<moirai-id>`, `<new-study>` подставить реальные имена:

```powershell
.\.venv\Scripts\python.exe -m tshybrid hybrid-report outputs/<core-id> outputs/<moirai-id> --output reports/<new-study> --reproduction reports/reproduction_corrected
```

`--reproduction` добавляет ранее выполненный контроль DLinear; он не запускает его заново.
Если этого каталога нет, параметр можно опустить. Нужны завершённые исходные прогоны
с прогнозами и сохранёнными компонентами гибрида, а не только `metrics.csv`.

### 7.4. Продолжение после частично выполненного сценария

Скрипт умеет пропускать целиком завершённый этап:

```powershell
.\.venv\Scripts\python.exe scripts/run_hybrid_study.py --resume-core outputs/<completed-core-id> --output reports/<new-study>
```

Или использовать обе готовые директории:

```powershell
.\.venv\Scripts\python.exe scripts/run_hybrid_study.py --resume-core outputs/<completed-core-id> --resume-moirai outputs/<completed-moirai-id> --output reports/<new-study>
```

Это не продолжение с середины обучения DLinear. `--resume-*` проверяет статус
этапа, а генератор отчёта дополнительно проверяет совместимость данных и origins.

### 7.5. Полные прогоны

Полные конфиги подготовлены, но вся матрица ещё не выполнена:

```powershell
.\.venv\Scripts\python.exe -m tshybrid run configs/benchmark.yaml --models last_value seasonal_naive ar arima ets dlinear chronos timesfm arima_chronos
.\.venv-moirai\Scripts\python.exe -m tshybrid run configs/benchmark.yaml --models moirai
```

В `benchmark.yaml` ARIMA оставлена с `initialization: default`; на Weather возможен
известный численный сбой. Для нового полного эксперимента следует явно подготовить
согласованную копию с выбранной инициализацией, например как в гибридных конфигах.
Не смешивать в одной таблице результаты разных настроек, принимая их за одну базу.
Оценка ARIMA на каждом окне и канале может быть очень долгой; GPU её не ускоряет автоматически.

## 8. Контроль воспроизведения DLinear

```powershell
.\.venv\Scripts\python.exe -m tshybrid run configs/reproduce_ettm1_h48.yaml
.\.venv\Scripts\python.exe -m tshybrid compare outputs/<run-id> --references references/hn_mvts_author.csv
.\.venv\Scripts\python.exe -m tshybrid compare outputs/<run-id> --references references/hn_mvts_paper.csv
```

После исправления окон проверенный запуск:

- run ID `20261026T185131Z-3d48c449`;
- все 7 каналов, первые 57 600 точек, L=336, H=48, 50 эпох, seed=0;
- 11 473 test origins, лучшая эпоха `epoch_id=41`;
- MSE **0.270279640964054**, авторский лог **0.270279648217062**.

Разница около `-7.253 × 10^-9`. Это воспроизведение одного авторского запуска.
В статье указаны mean/std по пяти seed; пять запусков здесь ещё не выполнены.
В старом контрольном прогоне `0.270063` была ошибочно усечена сетка окон; его
нельзя использовать как подтверждение репликации. Он сохранён как история.

Эталоны `references/*.csv` уже находятся в проекте. Заново извлечь авторские
данные можно при наличии полного соседнего HN-MVTS, включая упоминаемые в логах наборы:

```powershell
.\.venv\Scripts\python.exe -m tshybrid import-hn-reference --repo ../HN-MVTS
```

Статусы `within_tolerance`/`outside_tolerance` относятся к числам только при
совпадении проверяемого протокола. `protocol_mismatch` и `not_run` — отдельные случаи.

## 9. Где лежат результаты

### 9.1. Один эксперимент

`outputs/<run-id>/` содержит:

| Файл | Назначение |
|---|---|
| `report.html` | Просмотр метрик и ошибок запуска |
| `metrics.csv`, `aggregate.csv` | Итоговые оценки, агрегация по seed |
| `manifest.json`, `resolved.yaml` | Данные, версии, устройство, фактический конфиг |
| `selection.json` | Все validation-кандидаты и численные отказы |
| `failures.json` | Ошибки отдельных задач |
| `source.zip` | Snapshot исходников пакета на момент запуска |
| `<dataset>-<model>-H<h>-s<seed>/predictions.parquet` | Прогнозы по origin/channel/lead |
| `by_lead.csv`, `origins.npz` внутри задачи | Качество по шагам и точные позиции окон |
| `best.pt`, `training-*.csv` у DLinear | Веса лучшей эпохи и обучение |

Прочитать таблицу можно без Jupyter:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; print(pd.read_csv('reports/hybrid_study/comparisons.csv').to_string(index=False))"
```

### 9.2. Исследовательский отчёт

`reports/hybrid_study/` содержит HTML и `REPORT.md`, сравнения MSE/MAE/MASE,
интервалы, время, разбор по каналам/origins/lead, графики, примеры и provenance.
`moirai_point_diagnostics.csv` сравнивает исходное среднее samples с сохранённой
медианой отдельно от основной таблицы; `largest_errors.csv` показывает выбросы.

Исходные расширенные прогоны:

- core: `outputs/20261026T183514Z-f78629b8`;
- Moirai: `outputs/20261026T184059Z-cd8a44b0`.

Переиздать отчёт из них можно через `hybrid-report` в новый каталог, без нового inference.
После клонирования только git эти директории могут отсутствовать: для такого
переиздания потребуется восстановить raw outputs или повторить эксперименты.

### 9.3. Перестроение обычного HTML и объединение таблиц

```powershell
.\.venv\Scripts\python.exe -m tshybrid report outputs/<run-id> --references references/hn_mvts_author.csv
.\.venv\Scripts\python.exe -m tshybrid combine outputs/<first-id> outputs/<second-id> --output outputs/<new-combined-name>
```

`combine` проверяет данные и origins, отклоняет дубли model/seed.
`report` и `compare` могут обновлять производные файлы в указанной директории;
для сохранения истории сначала использовать копию. `combine` и `hybrid-report`
требуют нового выходного каталога.

## 10. Настройка собственного эксперимента

Скопировать ближайший YAML в тот же `configs/` и изменить:

- `datasets`: путь, split, `take_n`, список `channels`, сезонность;
- `context`, `horizons`: длина истории и горизонты в шагах;
- `evaluation.val_limit/test_limit`: ограничения пилота; убрать для полной сетки;
- `models`: `type`, уникальный `id`, `params`, validation `candidates`, `seeds`;
- `device`, `threads`, размеры батчей;
- `model_lock`: файл зафиксированных весов.

`split` всегда `[train, validation, test]`. `channels` — индексы, начиная с нуля.
Пути данных/output/cache разрешаются относительно директории конечного YAML, включая
`extends`. Словари наследуются, списки заменяются целиком.

Гибрид должен ссылаться на базу и FM, объявленные раньше в списке. Для AR/ARIMA
`fit_prefix` должен равняться `warmup`, а `warmup < context`. Зависимости гибрида
должны иметь один однозначный seed. Нельзя уменьшить context до 96, оставив warmup=128.

## 11. Проверки

```powershell
.\.venv\Scripts\python.exe -m pytest -q --disable-warnings
.\.venv\Scripts\ruff.exe check src tests scripts
```

После исправления окон выполнено 19 тестов. Без соседнего HN-MVTS тест сравнения
его forward с нашей реализацией будет пропущен. Предупреждения зависимостей и
`ConvergenceWarning` следует отличать от падения теста или неполного эксперимента.

## 12. Частые проблемы

| Симптом | Действие |
|---|---|
| `No module named tshybrid` | Установить локальный проект `-e .` в том окружении, чей python запускается |
| Нет `uni2ts` или конфликт NumPy/PyArrow | Запускать Moirai через `.venv-moirai`, восстановить её lock-файл |
| Не найден NPY | Подготовить данные; проверить разрешение пути относительно YAML |
| Ошибка download/SSL | Проверить доступ к Hugging Face; при доверенном системном CA использовать `--system-certs` |
| CUDA недоступна | Использовать CPU либо отдельно установить совместимую GPU-сборку PyTorch |
| `LU decomposition error` | Проверить ARIMA initialization; для расширенного пилота уже задана approximate_diffuse |
| ARIMA не сошлась, но прогноз сохранён | Смотреть `arima_nonconverged_fits` и диагностику отчёта; это ограничение надёжности результата |
| MSE Moirai очень велико | Посмотреть квантили и `largest_errors.csv`; не удалять выбросы и не заменять основной функционал задним числом |
| `Report directory already exists` | Передать новое имя через `--output` |
| `Different data/protocol/origins` | Сверить конфиги; результаты с разными каналами/окнами не являются одной сравнительной сеткой |
| `protocol_mismatch` для статьи | Проверить dataset SHA, окна, seed и настройки; совпавшее число при другом протоколе не доказывает репликацию |
| `No complete windows` | Увеличить длину split или уменьшить context/horizon в согласованном конфиге |

Успешный запуск — проверка работоспособности и конкретного эксперимента.
Воспроизведение статьи и доказательство устойчивого научного выигрыша требуют
дополнительных проверок, перечисленных в [SUPERVISOR_TASK_REVIEW.md](SUPERVISOR_TASK_REVIEW.md).
