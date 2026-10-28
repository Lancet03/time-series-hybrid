# Воспроизведение и ограничения

Источник: [HN-MVTS, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39711/43672),
Experimental setup и Table 2; [авторский код](https://github.com/av-savchenko/HN-MVTS).

`references/hn_mvts_author.csv` — **один seed** из авторских Parquet-логов, эпоха
выбрана по validation. Записаны имя/SHA исходного лога и SHA данных. Команда
`import-hn-reference` повторно извлекает эталоны из соседнего репозитория.

`references/hn_mvts_paper.csv` — опубликованные MSE mean/std по пяти seed для
DLinear на стартовых датасетах. Этот эталон не подменяется одиночным авторским логом.
Абсолютный допуск 0.005 для диагностики логов задан до запуска; он не является
критерием статистической эквивалентности.

## Профили

`common`: полные допустимые окна, element-weighted метрики. Это основа общего
сравнения; она отличается от особенностей старого кода.

`hn_mvts_code` сохраняет авторский режим:

- используются все допустимые validation/test окна, как в авторском коде;
- validation по эпохам использует shuffle=True и среднее метрик батчей;
- test MSE/MAE — невзвешенное среднее по батчам; micro-метрики доступны отдельно;
- affine RevIN, context=336, lr=0.0001, batch=64, 50 эпох;
- ETT: первые 57600 точек, split 6:2:2; Weather/ECL: split 7:1:2;
- сохраняется расход torch RNG на итератор авторского test-loader, но test
  не читается во время обучения и выбора checkpoint.

Пять сетей теперь реализованы в собственных модулях `models/`, без импортов
HN-MVTS или vendored-копии. Архитектуры соответствуют Base-вариантам эталона;
при наличии соседнего checkout тесты сравнивают инициализацию, forward и градиенты.
Без него сети и основные тесты обучения также работают. Происхождение
архитектур и адаптированного кода указано в `THIRD_PARTY.md`.
CPU/GPU, версии библиотек и порядок операций могут давать численные различия.

Описанные ниже сохранённые результаты получены прежней отдельной реализацией
DLinear. Они не переписаны; новый источник сети явно указывается в metadata
новых запусков. Отчёт `hn_models_laptop` относится к промежуточной версии с
перенесённым кодом. Эквивалентность сетей сама по себе не подтверждает репликацию таблиц.

В статье не-ETT split указан 7:2:1, в коде 7:1:2; исходный YAML также содержит
другой lr и один seed. Поэтому paper reference имеет профиль
`hn_mvts_paper_unresolved`: похожее число не объявляется репликацией автоматически.
Нужно разрешить splits/окна/агрегацию, выполнить пять seed и сравнить агрегат.

### Исправление разбора окон

В первоначальной версии проекта ошибочно вычитался ещё один context из числа
validation/test origins режима `hn_mvts_code`. В HN-MVTS срез `XY` индексирует начало
всего окна, а origin равен этому индексу плюс context. Поэтому авторский срез
`XY[start-L:end-L-H+1]` соответствует origins `start .. end-H` включительно.
Авторские окна не усечены. Ошибка исправлена; добавлен тест прямого сопоставления
индексов, а проверка эталонов теперь проверяет и число test origins.
Ранний результат 0.270063 на усечённой сетке не подтверждает репликацию; история
сохранена в артефактах. Протокол `common` и результаты гибридов не изменились.

Повторный прогон `20261026T185131Z-3d48c449` после исправления: test MSE
0.270279640964054 против авторского 0.270279648217062, разница около -7.253×10⁻⁹.
Использованы все 11 473 test origins и лучшая эпоха 41. Артефакты сохранены в
[`reports/reproduction_corrected/`](../reports/reproduction_corrected/).

```powershell
.\.venv\Scripts\python.exe -m tshybrid run configs/reproduce_ettm1_h48.yaml
.\.venv\Scripts\python.exe -m tshybrid compare outputs/<run-id> --references references/hn_mvts_author.csv
.\.venv\Scripts\python.exe -m tshybrid compare outputs/<run-id> --references references/hn_mvts_paper.csv
```

Проверяются protocol/context/seed, число каналов/точек, SHA данных, splits,
epochs/lr/batch/RevIN и отсутствие pilot. Статусы: not_run, protocol_mismatch,
within_tolerance, outside_tolerance. Подматрица не считается полной репликацией.

## Фундаментальные модели

Адаптеры трёх FM проверены на общем пилоте. Числа оригинальных FM-статей пока
**не воспроизведены**: нужны отдельные конфиги точных benchmarks, checkpoints,
контекстов, горизонтов, origins и вероятностных метрик.
Версии TimesFM 2.5 / Chronos-Bolt / Moirai 1.1 нельзя сравнивать с таблицами другой
версии модели только по названию семейства.

Официальные реализации оценки:

- [Chronos scripts](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts)
  и [fev](https://github.com/autogluon/fev).
- [TimesFM](https://github.com/google-research/timesfm).
- [Moirai benchmarks](https://github.com/SalesforceAIResearch/uni2ts/tree/main/project/benchmarks).

Следующий контрольный шаг — выбрать по одному конкретному опубликованному результату
для используемых версий FM и перенести полный протокол. Одного числа в CSV недостаточно.
