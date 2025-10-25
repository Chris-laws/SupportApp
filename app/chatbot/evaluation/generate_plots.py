import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fonts/Encoding: Umlaute sicher anzeigen, kein TeX, Minus-Zeichen korrekt
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 11

BASE_DIR = Path("app/chatbot/evaluation").resolve()
OUTPUT_SPECS: Dict[str, str] = {
    "k_sweep": "plot_k_sweep.png",
    "reranker_effect": "plot_reranker_effect.png",
    "chunking_effect": "plot_chunking_effect.png",
    "corr_recall_f1": "plot_correlation_recall_f1.png",
    "corr_recall_factual": "plot_correlation_recall_factual.png",
    "model_quality": "plot_model_quality.png",
    "model_efficiency": "plot_model_efficiency.png",
    "stability": "plot_stability.png",
}

COLORS = {
    "recall": "#1f77b4",
    "ndcg": "#ff7f0e",
    "f1": "#2ca02c",
    "factual": "#d62728",
    "latency": "#7f7f7f",
    "tokens": "#0f3b82",
}

# Einheitliche Standardwerte für die Diagramme
plt.rcParams.update(
    {
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.color": "#c0c0c0",
        "grid.alpha": 0.4,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "figure.dpi": 160,
    }
)


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def save_figure(fig: plt.Figure, key: str) -> None:
    filename = OUTPUT_SPECS[key]
    output_path = BASE_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {filename}")


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        warn(f"Datei nicht gefunden: {path.name}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        warn(f"Konnte CSV nicht laden ({path.name}): {exc}")
        return None


def as01(series: pd.Series) -> pd.Series:
    """Bringt Scores robust in den Bereich [0,1] (wandelt 0–100% zu 0–1 um)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().gt(1.0).any():
        numeric = numeric / 100.0
    return numeric.clip(lower=0.0, upper=1.0)


def clamp_axes01(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)


def ensure_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> bool:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        warn(f"{context}: Fehlende Spalten: {', '.join(missing)}")
        return False
    return True


def safe_mean(df: Optional[pd.DataFrame], cols: Iterable[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for col in cols:
        if col not in df.columns:
            return None
    numeric = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all().all():
        return None
    return numeric.mean()


def plot_k_sweep() -> None:
    df = load_csv(BASE_DIR / "summary_runs.csv")
    if df is None or df.empty:
        warn("summary_runs.csv: Keine Daten – Plot wird übersprungen")
        return
    required = ["top_k", "recall_mean", "ndcg_mean", "keyword_f1_mean", "factual_rate"]
    if not ensure_columns(df, required, "summary_runs.csv"):
        warn("[SKIP] k-sweep: fehlende Spalten")
        return

    sdf = df.sort_values("top_k").copy()
    sdf["recall_mean"] = as01(sdf["recall_mean"])
    sdf["ndcg_mean"] = as01(sdf["ndcg_mean"])
    sdf["keyword_f1_mean"] = as01(sdf["keyword_f1_mean"])
    sdf["factual_rate"] = as01(sdf["factual_rate"])

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.set_title("k-Sweep: Einfluss der Top-k-Größe auf Retrieval- und Antwortmetriken")
    ax.set_xlabel("k (Top-k im Kontext)")
    ax.set_ylabel("Score [0–1]")

    plotted_any = False
    for column, label, color in [
        ("recall_mean", "Recall", COLORS["recall"]),
        ("ndcg_mean", "nDCG", COLORS["ndcg"]),
        ("keyword_f1_mean", "Keyword-F1", COLORS["f1"]),
        ("factual_rate", "Faktentreue", COLORS["factual"]),
    ]:
        series = sdf[column]
        if series.notna().any():
            ax.plot(sdf["top_k"], series, marker="o", label=label, color=color)
            plotted_any = True
    if not plotted_any:
        warn("[SKIP] k-sweep: keine verwertbaren Werte")
        plt.close(fig)
        return
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Metrik")
    save_figure(fig, "k_sweep")


def aggregate_run_metrics(path: Path) -> Optional[Dict[str, float]]:
    df = load_csv(path)
    metrics = safe_mean(df, ["keyword_f1", "factual_correct"])
    if metrics is None:
        return None
    return {
        "f1": float(as01(pd.Series([metrics["keyword_f1"]])).iloc[0]),
        "factual": float(as01(pd.Series([metrics["factual_correct"]])).iloc[0]),
    }


def plot_grouped_bars(
    data: Dict[str, Dict[str, float]],
    labels: Iterable[str],
    metric_order: Tuple[Tuple[str, str, str], ...],
    title: str,
    key: str,
) -> None:
    if not data:
        warn(f"[SKIP] {key}: keine Daten verfügbar")
        return

    labels = list(labels)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_title(title)
    ax.set_ylabel("Score [0–1]")

    x = np.arange(len(labels))
    width = 0.35
    plotted_any = False

    for idx, (metric_key, metric_label, color_key) in enumerate(metric_order):
        offsets = x + (idx - (len(metric_order) - 1) / 2) * width
        values = []
        for label in labels:
            value = data.get(label, {}).get(metric_key)
            if value is None or math.isnan(value):
                values.append(np.nan)
            else:
                values.append(value)
                plotted_any = True
        ax.bar(offsets, values, width=width, label=metric_label, color=COLORS[color_key])

    if not plotted_any:
        warn(f"[SKIP] {key}: keine gültigen Werte")
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Metrik")
    save_figure(fig, key)


def plot_reranker_effect() -> None:
    base_metrics = aggregate_run_metrics(BASE_DIR / "results_hybrid_k10_base.csv")
    ce_metrics = aggregate_run_metrics(BASE_DIR / "results_hybrid_k10_ce50.csv")
    if base_metrics is None or ce_metrics is None:
        print("[SKIP] reranker_effect: fehlende Daten oder Spalten")
        return
    plot_grouped_bars(
        {"Base": base_metrics, "CE@50": ce_metrics},
        ["Base", "CE@50"],
        (("f1", "Keyword-F1", "f1"), ("factual", "Faktentreue", "factual")),
        "Einfluss des Cross-Encoders (Reranker) auf Antwortqualität",
        "reranker_effect",
    )


def plot_chunking_effect() -> None:
    base_metrics = aggregate_run_metrics(BASE_DIR / "results_hybrid_k10_ce50.csv")
    chunk_metrics = aggregate_run_metrics(BASE_DIR / "results_hybrid_k10_ce50_chunk260.csv")
    if base_metrics is None or chunk_metrics is None:
        print("[SKIP] chunking_effect: fehlende Daten oder Spalten")
        return
    plot_grouped_bars(
        {"Basisindex": base_metrics, "Chunk 260": chunk_metrics},
        ["Basisindex", "Chunk 260"],
        (("f1", "Keyword-F1", "f1"), ("factual", "Faktentreue", "factual")),
        "Einfluss der Chunkgröße auf Antwortqualität und Faktentreue",
        "chunking_effect",
    )


def plot_correlation(y_col: str, y_label: str, title: str, key: str, color_key: str) -> None:
    frames = []
    for path in BASE_DIR.glob("results_hybrid_k*.csv"):
        df = load_csv(path)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        warn(f"[SKIP] {key}: keine CSV-Dateien gefunden")
        return

    combined = pd.concat(frames, ignore_index=True)
    if not ensure_columns(combined, ["recall@k", y_col], "results_hybrid_k*.csv"):
        warn(f"[SKIP] {key}: notwendige Spalten fehlen")
        return

    x = as01(combined["recall@k"])
    y = as01(combined[y_col])
    mask = x.notna() & y.notna()
    if mask.sum() == 0:
        warn(f"[SKIP] {key}: keine gültigen Datenpunkte")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel(y_label)
    ax.scatter(x[mask], y[mask], s=22, color=COLORS[color_key], edgecolors="black", linewidths=0.3)

    x_valid = x[mask]
    y_valid = y[mask]
    if mask.sum() >= 3 and x_valid.std() > 1e-6 and y_valid.std() > 1e-6:
        m, b = np.polyfit(x_valid, y_valid, 1)
        xs = np.linspace(0.0, 1.0, 50)
        ax.plot(xs, m * xs + b, linestyle="--", color=COLORS[color_key])

    clamp_axes01(ax)
    save_figure(fig, key)


def plot_model_quality() -> None:
    df = load_csv(BASE_DIR / "summary_models.csv")
    if df is None or df.empty:
        warn("[SKIP] model_quality: keine Daten")
        return
    if not ensure_columns(df, ["model", "keyword_f1_mean", "factual_rate"], "summary_models.csv"):
        warn("[SKIP] model_quality: Spalten unvollständig")
        return

    df = df.copy()
    df["keyword_f1_mean"] = as01(df["keyword_f1_mean"])
    df["factual_rate"] = as01(df["factual_rate"])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_title("Vergleich der Modellqualität (F1 vs. Faktentreue)")
    ax.set_ylabel("Score [0–1]")

    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    f1_values = df["keyword_f1_mean"]
    factual_values = df["factual_rate"]
    if f1_values.notna().sum() == 0 and factual_values.notna().sum() == 0:
        warn("[SKIP] model_quality: keine gültigen Werte")
        plt.close(fig)
        return

    ax.bar(x - width / 2, f1_values, width=width, label="Keyword-F1", color=COLORS["f1"])
    ax.bar(x + width / 2, factual_values, width=width, label="Faktentreue", color=COLORS["factual"])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Metrik")
    save_figure(fig, "model_quality")


def plot_model_efficiency() -> None:
    df = load_csv(BASE_DIR / "summary_models.csv")
    if df is None or df.empty:
        warn("[SKIP] model_efficiency: keine Daten")
        return
    if not ensure_columns(df, ["model", "time_mean", "tokens_out_mean"], "summary_models.csv"):
        warn("[SKIP] model_efficiency: Spalten unvollständig")
        return

    df = df.copy()
    latency = pd.to_numeric(df["time_mean"], errors="coerce")
    tokens_out = pd.to_numeric(df["tokens_out_mean"], errors="coerce")
    if latency.notna().sum() == 0 and tokens_out.notna().sum() == 0:
        warn("[SKIP] model_efficiency: keine gültigen Werte")
        return

    models = df["model"].tolist()
    x = np.arange(len(models))

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.set_title("Effizienzvergleich der Modelle (Latenz und Token-Ausgabe)")
    ax1.set_ylabel("Latenz (Sekunden)")

    ax1.bar(x, latency, color=COLORS["latency"], alpha=0.8, label="Latenz")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15)

    ax2 = ax1.twinx()
    ax2.plot(x, tokens_out, color=COLORS["tokens"], marker="o", label="Tokens out")
    ax2.set_ylabel("Tokens")

    latency_handle = plt.Line2D([0], [0], color=COLORS["latency"], linewidth=6)
    tokens_handle = plt.Line2D([0], [0], color=COLORS["tokens"], marker="o", linewidth=2)
    ax1.legend([latency_handle, tokens_handle], ["Latenz", "Tokens out"], loc="upper left")
    save_figure(fig, "model_efficiency")


def plot_stability() -> None:
    df = load_csv(BASE_DIR / "summary_stability.csv")
    if df is None or df.empty:
        warn("[SKIP] stability: keine Daten")
        return
    if not ensure_columns(df, ["label", "keyword_f1_mean", "keyword_f1_std"], "summary_stability.csv"):
        warn("[SKIP] stability: Spalten unvollständig")
        return

    df = df.copy()
    df["keyword_f1_mean"] = as01(df["keyword_f1_mean"])
    df["keyword_f1_std"] = pd.to_numeric(df["keyword_f1_std"], errors="coerce")

    if df["keyword_f1_mean"].notna().sum() == 0:
        warn("[SKIP] stability: keine gültigen Werte")
        return

    labels = df["label"].tolist()
    x = np.arange(len(labels))
    means = df["keyword_f1_mean"]
    stds = df["keyword_f1_std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_title("Stabilität der Ergebnisse (Mittelwert ± Standardabweichung)")
    ax.set_ylabel("Keyword-F1")
    ax.bar(x, means, yerr=stds, color=COLORS["f1"], capsize=6, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0.0, 1.0)
    save_figure(fig, "stability")


def main() -> None:
    plot_k_sweep()
    plot_reranker_effect()
    plot_chunking_effect()
    plot_correlation(
        y_col="keyword_f1",
        y_label="Keyword-F1",
        title="Korrelation zwischen Retrievalqualität und Antwortqualität",
        key="corr_recall_f1",
        color_key="f1",
    )
    plot_correlation(
        y_col="factual_correct",
        y_label="Faktentreue",
        title="Korrelation zwischen Retrievalqualität und Faktentreue",
        key="corr_recall_factual",
        color_key="factual",
    )
    plot_model_quality()
    plot_model_efficiency()
    plot_stability()


if __name__ == "__main__":
    main()
