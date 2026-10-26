"""Paired base / raw FM / residual hybrid study, built from actual saved forecasts."""
from html import escape
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd

from .reporting import combine_runs


KEYS = ["origin", "channel", "lead"]
LABELS = {"chronos": "Chronos-Bolt Tiny", "timesfm": "TimesFM 2.5 200M", "moirai": "Moirai 1.1 Small"}


def paired_predictions(base, raw_fm, hybrid):
    """Reject missing/extra targets and verify the saved additive decomposition."""
    frames = []
    for frame in (base, raw_fm, hybrid):
        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate origin/channel/lead in forecasts")
        frames.append(frame.set_index(KEYS).sort_index())
    base, raw_fm, hybrid = frames
    if not base.index.equals(raw_fm.index) or not base.index.equals(hybrid.index):
        raise ValueError("Different prediction targets in the comparison triplet")
    required = {"base_prediction_scaled", "correction_scaled", "base_prediction", "correction"}
    if not required.issubset(hybrid.columns):
        raise ValueError("Hybrid artifact lacks saved base/correction components; rerun with current code")
    for column in ("y", "y_scaled"):
        np.testing.assert_allclose(base[column], raw_fm[column], rtol=1e-6, atol=1e-6,
                                   err_msg="Different targets/scaling in paired runs")
        np.testing.assert_allclose(base[column], hybrid[column], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(base.prediction_scaled, hybrid.base_prediction_scaled,
                               rtol=1e-5, atol=1e-6, err_msg="Hybrid uses a different base forecast")
    np.testing.assert_allclose(hybrid.prediction_scaled,
                               hybrid.base_prediction_scaled+hybrid.correction_scaled, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hybrid.prediction, hybrid.base_prediction+hybrid.correction,
                               rtol=1e-5, atol=1e-6)
    output = base[["y", "y_scaled"]].copy()
    for name, frame in (("base", base), ("fm", raw_fm), ("hybrid", hybrid)):
        output[f"{name}_prediction"] = frame.prediction
        output[f"{name}_prediction_scaled"] = frame.prediction_scaled
        error = frame.prediction_scaled.astype(float) - frame.y_scaled.astype(float)
        output[f"{name}_mse"] = error**2
        output[f"{name}_mae"] = np.abs(error)
    output["correction"] = hybrid.correction
    return output.reset_index()


def gain(reference, actual):
    return 100 * (reference-actual) / reference if reference > 0 else np.nan


def markdown_table(frame):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}" if np.isfinite(value) else "—"
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    rows += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def study_report(run_dirs, output, reproduction=None):
    from .runner import write_json
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(output / "report-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(__file__).parent.rglob("*.py")):
            archive.write(path, "tshybrid/" + path.relative_to(Path(__file__).parent).as_posix())
    run_dirs = [Path(p).resolve() for p in run_dirs]
    combine_runs(run_dirs, output / "all_models")
    rows, metrics_by_task, model_specs, manifests = [], {}, {}, []
    dataset_rows, model_rows, dataset_details = {}, [], {}
    for directory in run_dirs:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifests.append(manifest)
        for dataset in manifest["datasets"]:
            dataset_details[dataset["name"]] = dataset
            dataset_rows[dataset["name"]] = {
                "dataset": dataset["name"], "steps": dataset["used_shape"][0],
                "channels": dataset["used_shape"][1], "frequency": dataset.get("frequency"),
                "train": dataset["train_end"], "validation": dataset["val_end"]-dataset["train_end"],
                "test": dataset["test_end"]-dataset["val_end"], "seasonality": dataset["seasonality"],
                "selected_channels": json.dumps(dataset["channel_names"]), "sha256": dataset["sha256"]}
        specs = {spec["id"]: spec for spec in manifest["config"]["models"]}
        for _, row in pd.read_csv(directory / "metrics.csv").iterrows():
            key = (row.dataset, row.model, int(row.horizon), int(row.seed))
            metrics_by_task[key] = (row, directory)
            model_specs[key] = specs[row.model]
        for selected in manifest["models"]:
            model_rows.append({"run_id": manifest["run_id"], "dataset": selected["dataset"],
                               "model": selected["model"], "horizon": selected["horizon"],
                               "seed": selected["seed"], "validation_mse": selected["validation_mse"],
                               "selected_params": json.dumps(selected["selected_params"], sort_keys=True),
                               "metadata": json.dumps(selected["metadata"], sort_keys=True)})
        provenance = output / "provenance" / manifest["run_id"]
        provenance.mkdir(parents=True)
        for filename in ("manifest.json", "resolved.yaml", "selection.json", "failures.json", "source.zip"):
            if (directory / filename).exists():
                shutil.copyfile(directory / filename, provenance / filename)
    origins_tables, channels_tables, lead_tables, point_diagnostics, outlier_tables = [], [], [], [], []
    examples = {}
    for key, (hybrid_row, directory) in metrics_by_task.items():
        spec = model_specs[key]
        if spec["type"] != "hybrid":
            continue
        dataset, model, horizon, seed = key
        base_row, base_dir = metrics_by_task[(dataset, spec["base"], horizon, seed)]
        fm_row, fm_dir = metrics_by_task[(dataset, spec["residual"], horizon, seed)]
        def task_dir(folder, identifier):
            return folder / f"{dataset}-{identifier}-H{horizon}-s{seed}"
        base = pd.read_parquet(task_dir(base_dir, spec["base"]) / "predictions.parquet")
        raw = pd.read_parquet(task_dir(fm_dir, spec["residual"]) / "predictions.parquet")
        hybrid = pd.read_parquet(task_dir(directory, model) / "predictions.parquet")
        paired = paired_predictions(base, raw, hybrid)
        info = {"dataset": dataset, "fm": spec["residual"], "horizon": horizon, "seed": seed}
        if model_specs[(dataset, spec["residual"], horizon, seed)]["type"] == "moirai":
            detail = dataset_details[dataset]
            scales = dict(zip(detail["channel_names"], detail["scale"]))
            for role, frame in (("fm", raw), ("hybrid", hybrid)):
                scale = frame.channel.astype(str).map(scales)
                median_error = (frame["q0.5"]-frame.y)/scale
                mean_error = frame.prediction_scaled-frame.y_scaled
                point_diagnostics.append({**info, "role": role,
                                          "mean_mse": float((mean_error**2).mean()),
                                          "median_mse": float((median_error**2).mean()),
                                          "mean_mae": float(mean_error.abs().mean()),
                                          "median_mae": float(median_error.abs().mean()),
                                          "max_abs_mean_error_scaled": float(mean_error.abs().max()),
                                          "max_abs_median_error_scaled": float(median_error.abs().max())})
                largest = frame.loc[mean_error.abs().nlargest(5).index].copy()
                largest["abs_error_scaled"] = mean_error.abs().loc[largest.index]
                outlier_tables.append(largest.assign(**info, role=role))
        origin_table = paired.groupby("origin")[[f"{name}_{metric}" for name in ("base", "fm", "hybrid")
                                                  for metric in ("mse", "mae")]].mean().reset_index()
        origin_table = origin_table.assign(**info)
        origins_tables.append(origin_table)
        channel_table = paired.groupby("channel")[["base_mse", "fm_mse", "hybrid_mse"]].mean().reset_index()
        channels_tables.append(channel_table.assign(**info))
        lead_table = paired.groupby("lead")[["base_mse", "fm_mse", "hybrid_mse"]].mean().reset_index()
        lead_tables.append(lead_table.assign(**info))
        result = {**info, "base_model": spec["base"], "hybrid_model": model,
                  "channels": int(hybrid_row.channels), "origins": int(hybrid_row.n_origins),
                  "warmup": spec["params"]["warmup"], "context": int(hybrid_row.context),
                  "base_run": base_row.run_id, "fm_run": fm_row.run_id, "hybrid_run": hybrid_row.run_id,
                  "origin_win_vs_base": float((origin_table.hybrid_mse < origin_table.base_mse).mean()),
                  "origin_win_vs_fm": float((origin_table.hybrid_mse < origin_table.fm_mse).mean())}
        with np.load(task_dir(base_dir, spec["base"]) / "origins.npz") as positions:
            base_fits = (len(positions["validation"])+len(positions["test"]))*int(base_row.channels)
        result["base_fits_validation_test"] = base_fits
        result["base_nonconverged_fits"] = int(base_row.get("arima_nonconverged_fits", 0))
        result["hybrid_nonconverged_fits"] = int(hybrid_row.get("arima_nonconverged_fits", 0))
        result["base_nonconverged_fraction"] = result["base_nonconverged_fits"]/base_fits
        result["arima_initialization"] = model_specs[(dataset, spec["base"], horizon, seed)].get(
            "params", {}).get("initialization", "default")
        for name, metric_row in (("base", base_row), ("fm", fm_row), ("hybrid", hybrid_row)):
            for metric in ("mse", "mae", "mase", "coverage_80", "interval_score_80", "width_80",
                           "prediction_seconds", "peak_rss_mb"):
                result[f"{name}_{metric}"] = metric_row.get(metric, np.nan)
            actual_mse = paired[f"{name}_mse"].mean()
            if not np.isclose(actual_mse, metric_row.mse, rtol=1e-6, atol=1e-7):
                raise ValueError(f"Saved forecasts disagree with summary MSE: {key}, {name}")
        for metric in ("mse", "mae", "mase"):
            for reference in ("base", "fm"):
                result[f"{metric}_gain_vs_{reference}_pct"] = gain(result[f"{reference}_{metric}"], result[f"hybrid_{metric}"])
        if hybrid_row.n_origins * hybrid_row.channels:
            result["fm_ms_per_series_origin"] = 1000*fm_row.prediction_seconds/(fm_row.n_origins*fm_row.channels)
            result["hybrid_ms_per_series_origin"] = 1000*hybrid_row.prediction_seconds/(hybrid_row.n_origins*hybrid_row.channels)
        rows.append(result)
        # Predeclared illustration: first origin, first channel; never selected for a large gain.
        first_origin = base.origin.min()
        first_channel = str(base.loc[base.origin.eq(first_origin), "channel"].iloc[0])
        example = paired[paired.origin.eq(first_origin) & paired.channel.astype(str).eq(first_channel)]
        examples[(dataset, spec["residual"], horizon)] = example
        example_dir = output / "examples"
        example_dir.mkdir(exist_ok=True)
        example.to_csv(example_dir / f"{dataset}-{spec['residual']}-H{horizon}.csv", index=False)
    if not rows:
        raise ValueError("No hybrid comparisons found in these runs")
    comparisons = pd.DataFrame(rows).sort_values(["dataset", "horizon", "fm"])
    by_origin = pd.concat(origins_tables, ignore_index=True)
    by_channel = pd.concat(channels_tables, ignore_index=True)
    by_lead = pd.concat(lead_tables, ignore_index=True)
    for filename, frame in (("comparisons.csv", comparisons), ("by_origin.csv", by_origin),
                            ("by_channel.csv", by_channel), ("by_lead.csv", by_lead),
                            ("datasets.csv", pd.DataFrame(dataset_rows.values())),
                            ("selected_models.csv", pd.DataFrame(model_rows))):
        frame.to_csv(output / filename, index=False)
    diagnostics = pd.DataFrame(point_diagnostics)
    if not diagnostics.empty:
        diagnostics.to_csv(output / "moirai_point_diagnostics.csv", index=False)
        pd.concat(outlier_tables, ignore_index=True).to_csv(output / "largest_errors.csv", index=False)
    counts = {"cases": len(comparisons), "datasets": comparisons.dataset.nunique(),
              "hybrids": comparisons.fm.nunique(),
              "mse_beats_base": int((comparisons.hybrid_mse < comparisons.base_mse).sum()),
              "mse_beats_fm": int((comparisons.hybrid_mse < comparisons.fm_mse).sum()),
              "mse_beats_both": int(((comparisons.hybrid_mse < comparisons.base_mse)
                                      & (comparisons.hybrid_mse < comparisons.fm_mse)).sum())}
    write_json(output / "summary.json", counts)
    reproduction_table = pd.DataFrame()
    if reproduction:
        original = Path(reproduction).resolve()
        source = original / "comparison.csv"
        if source.exists():
            reproduction_table = pd.read_csv(source)
            reproduction_table = reproduction_table[reproduction_table.status.ne("not_run")]
            reproduction_table.to_csv(output / "reproduction.csv", index=False)
    figures = _figures(output, comparisons, by_lead, examples)
    _write_documents(output, comparisons, pd.DataFrame(dataset_rows.values()), counts,
                     reproduction_table, figures, manifests, diagnostics)
    return output / "report.html"


def _figures(output, comparisons, by_lead, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, reference in zip(axes, ("base", "fm")):
        ratios = comparisons.assign(mse_ratio=comparisons.hybrid_mse/comparisons[f"{reference}_mse"])
        table = ratios.pivot(index=["dataset", "horizon"], columns="fm", values="mse_ratio")
        values = table.to_numpy()
        logarithms = np.log10(values)
        bound = max(.1, np.nanmax(np.abs(logarithms)))
        chart = ax.imshow(logarithms, cmap="RdYlGn_r", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks(range(len(table.columns)), table.columns)
        ax.set_yticks(range(len(table)), [f"{d}, H={h}" for d, h in table.index])
        ax.set_title(f"Hybrid MSE / {'ARIMA' if reference == 'base' else 'raw FM'} MSE (<1 is better)")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                ax.text(col, row, f"{values[row,col]:.2g}x", ha="center", va="center", fontsize=9)
        fig.colorbar(chart, ax=ax, shrink=.8, label="log10(MSE ratio)")
    fig.savefig(output / "mse-gains.png", dpi=170)
    plt.close(fig)
    paths.append(("mse-gains.png", "Отношение MSE гибрида к референсу: меньше 1 означает улучшение; цвет — log10 отношения."))
    horizon = int(comparisons.horizon.max())
    datasets, families = sorted(comparisons.dataset.unique()), sorted(comparisons.fm.unique())
    fig, axes = plt.subplots(len(datasets), len(families), figsize=(13, 3*len(datasets)), squeeze=False,
                             constrained_layout=True)
    for i, dataset in enumerate(datasets):
        for j, fm in enumerate(families):
            frame = by_lead[(by_lead.dataset == dataset) & (by_lead.fm == fm) & (by_lead.horizon == horizon)]
            ax = axes[i,j]
            for label, name in (("ARIMA", "base"), ("Raw FM", "fm"), ("Hybrid", "hybrid")):
                ax.plot(frame.lead, frame[f"{name}_mse"], label=label, linewidth=1.2)
            ax.set(title=f"{dataset} / {fm}, H={horizon}", xlabel="Lead", ylabel="MSE (scaled)")
            if (frame[["base_mse", "fm_mse", "hybrid_mse"]] > 0).all().all():
                ax.set_yscale("log")
                ax.set_ylabel("MSE (scaled, log)")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.savefig(output / "by-lead.png", dpi=160)
    plt.close(fig)
    paths.append(("by-lead.png", f"MSE по шагам прогноза, H={horizon}; все выбранные каналы и origins."))
    for dataset in datasets:
        fig, axes = plt.subplots(2, len(families), figsize=(13, 5), squeeze=False, constrained_layout=True)
        for j, fm in enumerate(families):
            frame = examples[(dataset, fm, horizon)]
            ax = axes[0,j]
            ax.plot(frame.lead, frame.y, color="black", label="Observed")
            for label, name in (("ARIMA", "base"), ("Raw FM", "fm"), ("Hybrid", "hybrid")):
                ax.plot(frame.lead, frame[f"{name}_prediction"], label=label)
            ax.set(title=f"{dataset} / {fm}", xlabel="Lead", ylabel="Original units")
            if j == 0:
                ax.legend(fontsize=7)
            axes[1,j].plot(frame.lead, frame.correction)
            axes[1,j].axhline(0, color="grey", linewidth=.7)
            axes[1,j].set(xlabel="Lead", ylabel="FM correction (original units)")
        filename = f"example-{dataset}.png"
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
        paths.append((filename, f"{dataset}: первый origin и первый канал, выбранные до анализа ошибок."))
    return paths


def _write_documents(output, comparisons, datasets, counts, reproduction, figures, manifests, point_diagnostics):
    columns = ["dataset", "horizon", "fm", "base_mse", "fm_mse", "hybrid_mse",
               "mse_gain_vs_base_pct", "mse_gain_vs_fm_pct"]
    table = comparisons[columns]
    text = (
        f"Выполнено {counts['cases']} сравнений на {counts['datasets']} датасетах для {counts['hybrids']} FM. "
        f"Гибрид улучшил ARIMA по MSE в {counts['mse_beats_base']} случаях, FM на исходном ряде — "
        f"в {counts['mse_beats_fm']}, обе компоненты — в {counts['mse_beats_both']}. "
        "Это описательные результаты выбранной подвыборки, не проверка статистической значимости.")
    contexts = ", ".join(map(str, sorted(comparisons.context.unique())))
    horizons = "/".join(map(str, sorted(comparisons.horizon.unique())))
    origins = ", ".join(map(str, sorted(comparisons.origins.unique())))
    warmups = ", ".join(map(str, sorted(comparisons.warmup.unique())))
    protocol = (
        f"Общий протокол common: контекст исходного ряда {contexts}, H={horizons}, test origins: {origins}. "
        "Настройки ARIMA выбираются по validation MSE. Параметры фиксируются по первым "
        f"{warmups} точкам каждого контекста, затем обновляется состояние без refit. FM получает L-W причинных "
        "одношаговых остатков. Итог = база + коррекция с весом 1. Ни критерий выбора, ни вес коррекции "
        "по test не подбирались. Точный конфиг каждого запуска сохранён в provenance/.")
    initialization = ", ".join(sorted(comparisons.arima_initialization.unique()))
    diagnostics = (
        f"Инициализация ARIMA: {initialization}, maxiter=100 в этих конфигах. Approximate diffuse явно "
        "задана после LU-сбоя стационарной инициализации на Weather. Численно неудачные validation-кандидаты "
        "записываются в selection.json; ошибка выбранной модели на test делает прогон неполным. "
        "Часть локальных fit может завершаться без сходимости оптимизатора: такие прогнозы сохранены, "
        "а счётчики приведены ниже. Это ограничивает надёжность ARIMA baseline и выводов о её улучшении. "
        "Счётчики относятся к выбранному порядку на validation+test, не ко всем кандидатам.")
    limitations = (
        "Все результаты получены на CPU. Electricity ограничен фиксированными 16 каналами; origins выбраны "
        "равномерно по split, а не исчерпывают весь test. Это расширенный pilot. Каналы и соседние ошибки "
        "не являются независимыми наблюдениями; доля выигравших origins описательная, без p-value. "
        "FM используют разные point forecasts (Chronos: медиана; TimesFM: native head; Moirai: среднее samples). "
        "Интервалы гибрида условны на point базы. Для H=96 Chronos-Bolt использует rollout сверх native H=64. "
        "Moirai запускается в отдельном окружении со своим paired ARIMA baseline. Pretraining overlap не проверен. "
        "ARIMA ограничена двумя порядками и fit-prefix=128; сравнение не охватывает все ARIMA/SARIMA-настройки. "
        "Результаты общего протокола не воспроизводят таблицы FM-статей.")
    cost_note = (
        "Prediction time включает подгонку локальной ARIMA и FM inference, но не запись Parquet. "
        "Память — RSS всего процесса, включая cached модели, не VRAM. Cold load, validation selection "
        "и inference нельзя складывать без уточнения режима; метрики затрат здесь диагностические.")
    date = pd.Timestamp.now(tz="Europe/Moscow").strftime("%Y-%m-%d")
    md = [f"# Пункты 5 и 6: гибрид и первый отчёт ({date})", "", text, "", "## Протокол", "", protocol,
          "", markdown_table(datasets.drop(columns=["sha256", "selected_channels"])), "", "## MSE и выигрыш", "",
          markdown_table(table), "", "Положительный gain (%) означает снижение ошибки гибридом.",
          "", "## Полные результаты", "",
          "[comparisons.csv](comparisons.csv): MSE/MAE/MASE, интервалы, время и доли выигравших origins. "
          "[by_origin.csv](by_origin.csv), [by_channel.csv](by_channel.csv), [by_lead.csv](by_lead.csv). "
          "[selected_models.csv](selected_models.csv): выбранные настройки и checkpoints.", "", "## Стоимость", "", cost_note,
          "", "## Диагностика ARIMA", "", diagnostics,
          "", markdown_table(comparisons[["dataset", "horizon", "fm", "base_fits_validation_test", "base_nonconverged_fits", "hybrid_nonconverged_fits"]]),
          "", "## Ограничения", "", limitations]
    if not reproduction.empty:
        fields = [c for c in ["dataset", "horizon", "expected_mse", "actual_mse", "difference", "status"] if c in reproduction]
        md += ["", "## Ранее выполненный контроль DLinear", "", markdown_table(reproduction[fields]), "",
               "Сопоставление с авторским логом одного seed, не полная репликация Table 2 или FM-статей."]
    sensitivity_note = (
        "В основной таблице Moirai использует среднее 100 samples, как задано до запуска. "
        "В отдельных прогнозах среднее значительно выходит за центральный 80% интервал и даёт крупные ошибки. "
        "Ниже отдельно пересчитаны ошибки сохранённой q0.5 без нового inference и изменения основного результата. "
        "Это post-hoc диагностика чувствительности к точечному функционалу, не выбор модели по test. "
        "Крупнейшие ошибки и их квантили сохранены в largest_errors.csv; примеры выбраны именно для диагностики выбросов.")
    if not point_diagnostics.empty:
        md += ["", "## Moirai: среднее и медиана", "", sensitivity_note, "", markdown_table(point_diagnostics)]
    for filename, caption in figures:
        md += ["", f"![{caption}]({filename})", "", caption]
    md += ["", "## Как повторить", "", "```powershell",
           r".\.venv\Scripts\python.exe -m tshybrid run configs/hybrid_core.yaml",
           r".\.venv-moirai\Scripts\python.exe -m tshybrid run configs/hybrid_moirai.yaml",
           r".\.venv\Scripts\python.exe -m tshybrid hybrid-report outputs/<core-id> outputs/<moirai-id> --output reports/<new-study> --reproduction reports/reproduction",
           "```", "", "Сначала завершить основной запуск, затем Moirai, чтобы не конкурировать за CPU.",
           "", "Полные прогнозы находятся в outputs/<run-id>/. В отчёте хранятся компактные примеры и сводки; "
           "перед расчётом проверены одинаковые targets, совпадение базы и сумма base+correction."]
    (output / "REPORT.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    sections = ["<!doctype html><html lang='ru'><meta charset='utf-8'><title>Гибрид: первое исследование</title>",
                "<style>body{font:15px system-ui;max-width:1450px;margin:30px auto;padding:0 20px;line-height:1.5}"
                "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:7px}"
                "th{background:#edf3f8}img{max-width:100%;height:auto}.scroll{overflow:auto}"
                "pre{white-space:pre-wrap;background:#f4f6f8;padding:12px}</style>",
                f"<h1>Гибрид: пункты 5 и 6 · {date}</h1><p>{escape(text)}</p>",
                "<p><a href='REPORT.md'>Текст отчёта</a> · <a href='comparisons.csv'>Все сравнения (CSV)</a> · "
                "<a href='all_models/report.html'>Общая таблица запусков</a></p>",
                f"<h2>Протокол</h2><p>{escape(protocol)}</p>", datasets.drop(columns=["sha256", "selected_channels"]).to_html(index=False),
                "<h2>MSE: база / FM / гибрид</h2><p>Положительный gain (%) означает улучшение.</p>",
                "<div class='scroll'>"+table.to_html(index=False, float_format=lambda x: f"{x:.6g}")+"</div>"]
    for title, fields in (("MAE и MASE", ["dataset", "horizon", "fm", "base_mae", "fm_mae", "hybrid_mae", "base_mase", "fm_mase", "hybrid_mase"]),
                          ("Интервалы: номинальное покрытие 80%", ["dataset", "horizon", "fm", "fm_coverage_80", "hybrid_coverage_80", "fm_interval_score_80", "hybrid_interval_score_80"]),
                          ("Стоимость прогноза", ["dataset", "horizon", "fm", "base_prediction_seconds", "fm_prediction_seconds", "hybrid_prediction_seconds", "fm_ms_per_series_origin", "hybrid_ms_per_series_origin"]),
                          ("Доля origins со снижением MSE", ["dataset", "horizon", "fm", "origin_win_vs_base", "origin_win_vs_fm"])):
        sections += [f"<h2>{title}</h2>", "<div class='scroll'>"+comparisons[fields].to_html(index=False, float_format=lambda x: f"{x:.6g}")+"</div>"]
    sections += [f"<p>{escape(cost_note)}</p>", "<h2>Ограничения</h2>", f"<p>{escape(limitations)}</p>"]
    sections += ["<h2>Диагностика ARIMA</h2>", f"<p>{escape(diagnostics)}</p>",
                 comparisons[["dataset", "horizon", "fm", "base_fits_validation_test", "base_nonconverged_fits", "hybrid_nonconverged_fits"]].to_html(index=False)]
    if not reproduction.empty:
        fields = [c for c in ["dataset", "horizon", "expected_mse", "actual_mse", "status"] if c in reproduction]
        sections += ["<h2>Предыдущий контроль DLinear</h2>", reproduction[fields].to_html(index=False),
                     "<p>Авторский лог одного seed. Полное воспроизведение статей не заявляется.</p>"]
    if not point_diagnostics.empty:
        sections += ["<h2>Moirai: чувствительность среднего прогноза</h2>", f"<p>{escape(sensitivity_note)}</p>",
                     "<div class='scroll'>"+point_diagnostics.to_html(index=False, float_format=lambda x: f"{x:.6g}")+"</div>",
                     "<p><a href='largest_errors.csv'>Крупнейшие ошибки и квантили</a> · "
                     "<a href='moirai_point_diagnostics.csv'>Диагностика CSV</a></p>"]
    for filename, caption in figures:
        sections += [f"<figure><img src='{filename}' alt='{escape(caption)}'><figcaption>{escape(caption)}</figcaption></figure>"]
    sections += ["<h2>Воспроизводимость</h2><p>Конфиги, validation selection, checkpoints, пакеты и snapshot кода сохранены в provenance/.</p>",
                 "<pre>"+escape("\n".join(m["run_id"] for m in manifests))+"</pre>", "</html>"]
    (output / "report.html").write_text("\n".join(sections), encoding="utf-8")
