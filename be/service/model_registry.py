import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

import mlflow
from mlflow.tracking import MlflowClient

from be.config.settings import settings
from be.utils.logger import get_logger

logger = get_logger(__name__)

# config from reference
ARTIFACT_PATH = "model"
LOCAL_DOWNLOAD_DIR = "./model_cache"

MAX_PARALLEL_DOWNLOADS = 4

# populated at startup by download_all; run_id -> registration info
_registered: dict[str, dict] = {}


class ArtifactPaths(NamedTuple):
    artifact_dir: str
    metadata_path: str
    processor_dir: str
    model_dir: str


def _resolve_paths(run_id: str) -> ArtifactPaths:
    model_download_dir = os.path.join(LOCAL_DOWNLOAD_DIR, run_id)
    artifact_dir = os.path.join(model_download_dir, ARTIFACT_PATH)
    return ArtifactPaths(
        artifact_dir=artifact_dir,
        metadata_path=os.path.join(artifact_dir, "metadata.json"),
        processor_dir=os.path.join(artifact_dir, "processor"),
        model_dir=os.path.join(artifact_dir, "huggingface_model"),
    )


def _create_client() -> MlflowClient:
    os.environ["DATABRICKS_HOST"] = settings.DATABRICKS_HOST
    os.environ["DATABRICKS_TOKEN"] = settings.DATABRICKS_TOKEN
    mlflow.set_tracking_uri("databricks")
    return MlflowClient()


def clean_run_ids(raw_run_ids: list[str]) -> list[str]:
    return [rid.strip() for rid in raw_run_ids if rid and rid.strip()]


def _ensure_artifact(client: MlflowClient, run_id: str) -> ArtifactPaths:
    """
    Download artifacts for one run unless metadata.json is already cached.
    """

    paths = _resolve_paths(run_id)

    if os.path.exists(paths.metadata_path):
        logger.info(f"run_id {run_id} already exists locally. Skipping download.")
        return paths

    logger.info(f"Downloading run_id {run_id} from Databricks...")
    client.download_artifacts(
        run_id=run_id, path=ARTIFACT_PATH, dst_path=os.path.dirname(paths.artifact_dir)
    )
    return paths


def download_all(raw_run_ids: list[str]) -> dict[str, ArtifactPaths]:
    """
    Ensure local artifacts for every configured run id and register their
    metadata (name, quantization flag) for lazy loading and the catalog.

    Downloads run in parallel (I/O bound); any failure is logged and aborts
    startup after all workers finish so every error is visible at once.
    """

    run_ids = clean_run_ids(raw_run_ids)

    client = _create_client()

    if not run_ids:
        logger.info("No DATABRICKS_RUN_IDS configured; skipping artifact download.")
        return {}

    logger.info(
        f"Checking artifacts for {len(run_ids)} model(s) "
        f"(up to {min(MAX_PARALLEL_DOWNLOADS, len(run_ids))} parallel downloads)..."
    )

    ensured: dict[str, ArtifactPaths] = {}
    failures: list[tuple[str, Exception]] = []

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_DOWNLOADS, len(run_ids))) as pool:
        futures = {pool.submit(_ensure_artifact, client, rid): rid for rid in run_ids}
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                ensured[run_id] = future.result()
            except Exception as exc:
                failures.append((run_id, exc))
                logger.error(f"Artifact download failed for run_id {run_id}: {exc}")

    if failures:
        run_id, exc = failures[0]
        raise RuntimeError(
            f"{len(failures)} artifact download(s) failed; first failure: run_id {run_id}: {exc}"
        ) from exc

    paths_by_run_id: dict[str, ArtifactPaths] = {}
    for run_id in run_ids:
        paths = ensured[run_id]
        metadata = read_metadata(paths, run_id)

        _registered[run_id] = {
            "paths": paths,
            "metadata": metadata,
            "name": metadata.get("model_name", run_id) if metadata else run_id,
            "is_quantized": bool(metadata.get("is_quantized", False)) if metadata else False,
        }
        paths_by_run_id[run_id] = paths

    return paths_by_run_id


def read_metadata(paths: ArtifactPaths, run_id: str) -> dict | None:
    """
    Read metadata.json; warn instead of failing silently when unreadable.
    """

    if not os.path.exists(paths.metadata_path):
        logger.warning(f"metadata.json missing for run_id {run_id}; using defaults.")
        return None
    try:
        with open(paths.metadata_path, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Unreadable metadata.json for run_id {run_id}: {exc}")
        return None


def registered_models() -> dict[str, dict]:
    """
    Metadata for every artifact ensured this startup, keyed by run id.
    """

    return {
        rid: {"name": info["name"], "is_quantized": info["is_quantized"]}
        for rid, info in _registered.items()
    }


def get_paths(run_id: str) -> ArtifactPaths:
    """
    Local artifact paths for a registered run id.
    """

    return _registered[run_id]["paths"]
