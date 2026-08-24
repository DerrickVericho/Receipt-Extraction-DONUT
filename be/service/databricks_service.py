"""
Backward-compatible facade for callers that predate the module split.

Implementation lives in:
- model_registry.py : Databricks/MLflow artifact download, cache and metadata
- donut_service.py  : device setup, lazy model loading (LRU-capped), warmup, inference
- model_catalog.py  : experiment metrics, badges and model listing
"""

from be.service.donut_service import preload_model, run_extraction
from be.service.model_catalog import get_available_models

__all__ = ["preload_model", "run_extraction", "get_available_models"]
