import json
import os
import re

from be.config.settings import settings
from be.service import model_registry
from be.service.donut_service import ml_components
from be.utils.logger import get_logger

logger = get_logger(__name__)

_experiment_index: dict[str, dict] = {}

# badge/category thresholds
LOW_ACCURACY_F1 = 0.2
RECOMMENDED_F1_MIN = 0.78
RECOMMENDED_F1_DROP_MAX = 0.02
COMPACT_F1_MIN = 0.7


def load_experiment_results():
    """
    Load experiment results JSON, index by MLflow Run ID.
    Called once at startup after artifacts are registered.
    """

    global _experiment_index

    path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", settings.EXPERIMENT_RESULTS_PATH)
    )

    if not os.path.exists(path):
        logger.warning(f"Experiment results file not found at {path}. Metrics unavailable.")
        return

    with open(path, "r") as f:
        data = json.load(f)

    for row in data:
        run_id = row.get("MLflow Run ID")
        if run_id:
            _experiment_index[run_id] = row

    logger.info(
        f"Loaded experiment results for {len(_experiment_index)} models."
    )


def _has_metrics(row: dict) -> bool:
    return bool(row) and row.get("F1-Score") is not None


def _compute_badges(row: dict, model_name: str, is_quantized: bool) -> list[str]:
    """Compute display badges from experiment metrics."""

    badges = []

    if model_name == "DONUT-Base":
        badges.append("Highest accuracy")
    elif not _has_metrics(row):
        badges.append("Metrics unavailable")
    else:
        f1 = row.get("F1-Score") or 0
        f1_drop = row.get("F1 Drop") or 0

        if f1 < LOW_ACCURACY_F1:
            badges.append("Experimental · Low accuracy")
        elif is_quantized:
            badges.append("Quantized · INT8")
        elif f1 >= RECOMMENDED_F1_MIN and f1_drop <= RECOMMENDED_F1_DROP_MAX:
            badges.append("Recommended")
        else:
            badges.append("Compact")

    if "KD" in model_name:
        badges.append("Knowledge distilled")

    return badges


def _pipeline_sort_key(model: dict) -> tuple:
    """Base first, then pruning ratio asc, then pipeline stage (P → KD → KD-Q)."""
    
    name = model["name"] or ""
    if model["category"] == "base":
        return (0, 0, 0)

    ratio_match = re.search(r"P(\d+)", name)
    if not ratio_match:
        return (2, 0, 0)

    stage = 2 if name.endswith("-KD-Q") else 1 if name.endswith("-KD") else 0
    return (1, int(ratio_match.group(1)), stage)


def get_available_models() -> list[dict]:
    """
    Build the model catalog for every registered model (loaded or not),
    enriched with experiment metrics.
    """

    logger.info("Building available models catalog from registered models")
    available_models = []

    for run_id, info in model_registry.registered_models().items():
        row = _experiment_index.get(run_id, {})
        model_name = info.get("name", run_id)
        is_quantized = bool(info.get("is_quantized", False))
        has_metrics = _has_metrics(row)

        f1 = row.get("F1-Score") or 0
        f1_drop = row.get("F1 Drop") or 0
        size_red = row.get("Size Reduction (%)") or 0

        badges = _compute_badges(row, model_name, is_quantized)

        # category and summary
        if model_name == "DONUT-Base":
            category = "base"
            summary = f"Highest extraction quality. F1: {f1:.4f}"
        elif not has_metrics:
            category = "unranked"
            summary = "Experiment metrics unavailable for this model."
        elif f1 < LOW_ACCURACY_F1:
            category = "experimental"
            summary = f"Experimental. F1 score of {f1:.4f} — use with caution."
        elif is_quantized:
            category = "compact"
            summary = "Smaller INT8-quantized model. Uses less storage."
        elif f1_drop <= RECOMMENDED_F1_DROP_MAX:
            category = "balanced"
            summary = f"Near-base accuracy with {size_red:.1f}% smaller model. F1: {f1:.4f}"
        elif f1 >= COMPACT_F1_MIN:
            category = "compact"
            summary = f"F1: {f1:.4f}, {size_red:.1f}% smaller."
        else:
            category = "compact"
            summary = f"Reduced quality. F1: {f1:.4f}"

        available_models.append({
            "id": run_id,
            "name": model_name,
            "is_quantized": is_quantized,
            "loaded": run_id in ml_components,
            "recommended": "Recommended" in badges,
            "category": category,
            "summary": summary,
            "badges": badges,
            "metrics": {
                "precision": row.get("Precision"),
                "recall": row.get("Recall"),
                "f1_score": row.get("F1-Score"),
                "n_ted": row.get("N-TED"),
                "size_mb": row.get("Size (MB)"),
                "latency_ms": row.get("Latency (ms/sample)"),
                "flops_gflops": row.get("FLOPs (GFLOPs)"),
            } if row else {},
            "comparison_to_base": {
                "size_reduction_percent": row.get("Size Reduction (%)"),
                "latency_reduction_percent": row.get("Latency Reduction (%)"),
                "f1_drop": row.get("F1 Drop"),
            } if row else {},
        })

    available_models.sort(key=_pipeline_sort_key)

    return available_models
