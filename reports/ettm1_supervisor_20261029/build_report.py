"""Build presentation artifacts from completed runs; never starts model training."""
from pathlib import Path
import base64
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
summary = pd.read_csv(ROOT / "summary.csv").set_index(["model", "horizon"])
metrics = pd.read_csv(ROOT / "data" / "metrics.csv")
assert len(metrics) == 62
assert not metrics.duplicated(["dataset", "model", "horizon", "seed"]).any()
assert metrics["pilot"].eq(False).all()
assert metrics["dataset"].eq("ETTm1").all()
assert set(metrics["horizon"]) == {48, 96}
assert json.loads((ROOT / "data" / "failures.json").read_text()) == []

names = {
    "ar": "AR", "arima": "ARIMA", "dlinear": "DLinear",
    "tsmixer": "TSMixer", "moderntcn": "ModernTCN", "patchtst": "PatchTST",
    "itransformer": "iTransformer", "timesfm": "TimesFM",
    "chronos": "Chronos", "moirai": "Moirai", "arima_chronos": "ARIMA + Chronos",
}
order = list(names)
colors = {m: "#2563eb" for m in order}
colors.update(arima="#64748b", chronos="#059669", timesfm="#059669",
              moirai="#059669", arima_chronos="#ea580c")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# Keep the divergent AR result in the table, and explicitly exclude it here.
plot_order = sorted([m for m in order if m != "ar"],
                    key=lambda m: summary.loc[(m, 48), "mse_mean"])
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for ax, h in zip(axes, [48, 96]):
    for i, m in enumerate(plot_order):
        r = summary.loc[(m, h)]
        ax.errorbar(r.mse_mean, i, xerr=None if pd.isna(r.mse_std) else r.mse_std,
                    fmt="o", color=colors[m], capsize=4, markersize=7)
        ax.annotate(f"{r.mse_mean:.4f}", (r.mse_mean, i), xytext=(8, 5),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.22, 2.6)
    ax.set_xticks([.25, .5, 1, 2], ["0.25", "0.5", "1.0", "2.0"])
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.set_title(f"Горизонт {h} ({h // 4} ч)")
    ax.set_xlabel("MSE, логарифмическая шкала; меньше — лучше")
    ax.grid(axis="x", alpha=.2, which="both")
axes[0].set_yticks(range(len(plot_order)), [names[m] for m in plot_order])
axes[0].invert_yaxis()
fig.suptitle("ETTm1: качество прогноза на тестовой части", fontsize=15)
fig.text(.5, .015, "AR исключён с графика из-за расходимости; он приведён в таблице. "
         "Усы: ±1 SD по 5 seed; остальные точки — один запуск.", ha="center", fontsize=9)
fig.tight_layout(rect=(0, .05, 1, .95))
fig.savefig(ROOT / "mse_comparison.png", dpi=180)
fig.savefig(ROOT / "mse_comparison.svg")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, h in zip(axes, [48, 96]):
    models = ["arima", "chronos", "arima_chronos"]
    values = [summary.loc[(m, h), "mse_mean"] for m in models]
    bars = ax.bar([names[m] for m in models], values, color=[colors[m] for m in models])
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=4)
    ax.set_title(f"Горизонт {h}")
    ax.set_ylim(0, .82)
    ax.grid(axis="y", alpha=.2)
axes[0].set_ylabel("MSE; меньше — лучше")
fig.suptitle("Гибрид улучшает ARIMA, но уступает Chronos", fontsize=14)
fig.tight_layout()
fig.savefig(ROOT / "hybrid_comparison.png", dpi=180)
fig.savefig(ROOT / "hybrid_comparison.svg")
plt.close(fig)

def fmt(model, h, metric):
    row = summary.loc[(model, h)]
    value, std = row[f"{metric}_mean"], row[f"{metric}_std"]
    if model == "ar":
        return f"{value:.3e}"
    return f"{value:.4f}" + (f" ± {std:.4f}" if pd.notna(std) else "")

header = ["Модель", "MSE, H=48", "MAE, H=48", "MSE, H=96", "MAE, H=96"]
rows = [[names[m] + (" ⚠" if m == "ar" else ""),
         fmt(m, 48, "mse"), fmt(m, 48, "mae"),
         fmt(m, 96, "mse"), fmt(m, 96, "mae")] for m in order]
table_md = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
table_md += "\n".join("| " + " | ".join(row) + " |" for row in rows)
table_html = "<table><thead><tr>" + "".join(f"<th>{x}</th>" for x in header)
table_html += "</tr></thead><tbody>" + "".join(
    "<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in row) + "</tr>"
    for row in rows) + "</tbody></table>"

improvement = {}
for h in [48, 96]:
    a, c, hybrid = [summary.loc[(m, h), "mse_mean"]
                    for m in ["arima", "chronos", "arima_chronos"]]
    improvement[h] = ((a - hybrid) / a * 100, (hybrid / c - 1) * 100)

sections = [
    ("Что исследовали", [
        "Проверили качество статистических моделей, обучаемых нейросетей и замороженных "
        "фундаментальных моделей на ETTm1. Отдельная гипотеза: прогнозирование остатков "
        "ARIMA с помощью Chronos улучшает качество относительно каждой модели по отдельности.",
        "Все итоговые запуски завершены: 10 базовых архитектур и один гибрид, 62 строки "
        "метрик model/horizon/seed. Это предварительный результат на одном датасете, "
        "а не подтверждение точного воспроизведения всех экспериментов преподавателя."]),
    ("Протокол", [
        "ETTm1: 57 600 наблюдений, 7 каналов, интервал 15 минут. Хронологические части: "
        "train [0,34560), validation [34560,46080), test [46080,57600). Контекст 336 точек "
        "(84 часа), горизонты 48 и 96 (12 и 24 часа). Тестовые окна со сдвигом 1: "
        "11 473 и 11 425 начал прогноза соответственно. Параметры выбираются по validation.",
        "Масштабирование по среднему и стандартному отклонению только train, отдельно для "
        "каждого канала. Таблица показывает MSE/MAE в стандартизованной шкале; меньше — лучше. "
        "У DLinear, TSMixer, ModernTCN, PatchTST и iTransformer — среднее ± выборочное SD "
        "по пяти seed (0–4). У остальных моделей — один запуск. SD не является доверительным интервалом.",
        "Фундаментальные модели не дообучались. Checkpoint: amazon/chronos-bolt-tiny, "
        "google/timesfm-2.5-200m-pytorch, Salesforce/moirai-1.1-R-small. Зафиксированные "
        "версии моделей и параметры находятся в артефактах исходных запусков."]),
    ("Основные выводы", [
        "Лучшее среднее MSE: ModernTCN при H=48 (0.2583), PatchTST при H=96 (0.2877). "
        "При H=48 PatchTST очень близок к ModernTCN (0.2586); заявлять статистически "
        "значимое превосходство по этим средним нельзя.",
        "Лучший наблюдаемый MAE при обоих горизонтах — у TimesFM: 0.3115 и 0.3354. "
        "TimesFM также имеет лучшее MSE среди трёх исследованных фундаментальных моделей. "
        "Лидер зависит от выбранной метрики.",
        "AR показал численную расходимость на отдельных окнах: MSE порядка 10^13 и 10^34. "
        "Такой результат следует показывать как проблему устойчивости текущей реализации "
        "базовой модели. Он не позволяет делать общий вывод о непригодности AR."]),
    ("Проверка гибридной гипотезы", [
        "Прогноз гибрида = прогноз ARIMA + прогноз Chronos на остатках. Для обоих горизонтов "
        "выбран ARIMA(2,0,0) из двух кандидатов: (2,1,0) и (2,0,0). Коэффициенты оцениваются "
        "на первых 128 точках контекста, следующие 208 обрабатываются причинным фильтром "
        "с фиксированными параметрами; эти инновации подаются в Chronos.",
        f"Гибрид уменьшил MSE относительно ARIMA на {improvement[48][0]:.2f}% при H=48 "
        f"и {improvement[96][0]:.2f}% при H=96. Но его MSE выше, чем у отдельного Chronos, "
        f"на {improvement[48][1]:.1f}% и {improvement[96][1]:.1f}% соответственно. "
        "Гипотеза о преимуществе простой аддитивной комбинации над обеими базовыми моделями "
        "в данном эксперименте не подтвердилась.",
        "При H=48 зарегистрировано 2 768 несошедшихся подгонок ARIMA, при H=96 — 2 762 "
        "(около 1.7% подгонок выбранного кандидата на validation и test). Это диагностический "
        "счётчик оптимизации, а не число ошибок прогнозов и не показатель точности."]),
    ("Ограничения интерпретации", [
        "Один датасет и два горизонта. Пять повторов выполнены только для обучаемых нейросетей; "
        "значимость различий не проверена. Соседние тестовые окна перекрываются, поэтому их "
        "нельзя считать независимыми наблюдениями при проверке значимости.",
        "ARIMA использует ограниченный подбор порядков и короткий fit_prefix; её качество "
        "не является результатом исчерпывающей настройки ARIMA. Устойчивость AR требует исправления.",
        "Отдельный Chronos получает 336 исходных значений, а остаточная ветвь — 208 инноваций. "
        "Это различие контекстов нужно учитывать при сравнении гибрида. При H=96 Chronos "
        "работает за пределами родного горизонта 64 с авторегрессионным продолжением.",
        "Фундаментальные модели обрабатывают каналы по отдельности. Точечные прогнозы "
        "получены разными способами: медиана у Chronos, point head у TimesFM, среднее "
        "100 выборок у Moirai. Такие оценки по-разному связаны с MSE и MAE. Возможное "
        "пересечение предобучения с ETTm1 отдельно не проверялось.",
        "Базовые модели запускались параллельно, Moirai — в отдельном CPU-окружении. "
        "Сохранённые времена не подходят для честного ранжирования скорости архитектур."]),
    ("Что показать преподавателю", [
        "1. Постановку задачи и схему ARIMA → остатки → Chronos → сумма прогнозов.",
        "2. Протокол: данные, разбиение, контекст, горизонты, метрики и повторы.",
        "3. Полную таблицу всех 11 моделей и график MSE. Не скрывать расходимость AR.",
        "4. Отдельное сравнение ARIMA / Chronos / ARIMA + Chronos с относительными изменениями.",
        "5. Ограничения и следующий шаг: стабилизировать статистическую базу, проверить "
        "предсказуемость остатков, подобрать вес поправки только на validation, повторить "
        "на втором датасете. Это план следующих экспериментов, не выполненный результат."]),
]

md = "# ETTm1: результаты для обсуждения с преподавателем\n\nДата подготовки: 29.10.2026.\n\n"
for title, paragraphs in sections[:2]:
    md += f"## {title}\n\n" + "\n\n".join(paragraphs) + "\n\n"
md += "## Сравнительная таблица\n\n" + table_md + "\n\n"
md += "![Сравнение MSE](mse_comparison.png)\n\n"
for title, paragraphs in sections[2:]:
    md += f"## {title}\n\n" + "\n\n".join(paragraphs) + "\n\n"
    if title == "Проверка гибридной гипотезы":
        md += "![Сравнение гибрида](hybrid_comparison.png)\n\n"
md += ("## Файлы и происхождение результатов\n\n"
       "- `index.html` — автономный отчёт с встроенными графиками; можно отправить одним файлом.\n"
       "- `summary.csv` — сводная таблица; `summary_excel.csv` — вариант с `;` и десятичной запятой.\n"
       "- `data/metrics.csv` — все 62 строки исходных метрик, включая raw-шкалу.\n"
       "- `data/manifest.json` — девять исходных завершённых запусков и их идентификаторы.\n"
       "- `data/selection.json` — выбранные параметры.\n"
       "- PNG/SVG — графики для слайдов. `build_report.py` повторно строит отчёт из сохранённых CSV.\n\n"
       "ARIMA и Chronos взяты из совместного трёхмодельного эксперимента "
       "20261028T221119Z-ab32cd3e; остальные модели — из завершённых индивидуальных запусков. "
       "Совместимость данных, разбиений, контекстов и начал прогноза проверена при объединении. "
       "Старые неудачные попытки обучения не включены. Новое обучение при подготовке отчёта не запускалось.\n")
(ROOT / "REPORT.md").write_text(md, encoding="utf-8")

def img(filename, alt):
    data = base64.b64encode((ROOT / filename).read_bytes()).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'

body = "<h1>ETTm1: результаты исследования</h1><p>Материал для обсуждения с преподавателем · 29.10.2026</p>"
for i, (title, paragraphs) in enumerate(sections):
    body += f"<h2>{title}</h2>" + "".join(f"<p>{html.escape(p)}</p>" for p in paragraphs)
    if i == 1:
        body += "<h2>Сравнительная таблица</h2><div class='table'>" + table_html + "</div>"
        body += img("mse_comparison.png", "Сравнение MSE всех моделей кроме нестабильного AR")
    if title == "Проверка гибридной гипотезы":
        body += img("hybrid_comparison.png", "ARIMA, Chronos и гибрид: MSE")
body += ("<h2>Проверяемость</h2><p>62 строки исходных метрик, выбранные параметры и происхождение "
         "девяти завершённых запусков сохранены в каталоге data. ARIMA и Chronos взяты из "
         "совместного эксперимента 20261028T221119Z-ab32cd3e. Проверена совместимость данных, "
         "разбиений, контекстов и начал прогнозов. Новое обучение при подготовке отчёта не запускалось.</p>")
document = """<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ETTm1 — результаты для преподавателя</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#172033;background:#fff;max-width:1120px;margin:40px auto;padding:0 24px}
h1,h2{line-height:1.25}h1{font-size:32px}h2{margin-top:36px;font-size:23px}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #dbe2eb;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{background:#edf2f8}.table{overflow:auto}
img{max-width:100%;display:block;margin:24px auto}p{max-width:100ch}
@media print{body{margin:0;font-size:11px}h2{break-after:avoid}tr,img{break-inside:avoid}th,td{padding:5px}}
</style><body>""" + body + "</body></html>"
(ROOT / "index.html").write_text(document, encoding="utf-8")
print("Prepared REPORT.md, index.html, and four PNG/SVG charts.")
print("MSE changes versus ARIMA / versus Chronos:", improvement)
