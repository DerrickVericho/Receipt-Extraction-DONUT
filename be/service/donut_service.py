import io
import os
import re
import threading
import time
from collections import OrderedDict

import torch
from transformers import (
    VisionEncoderDecoderModel,
    AutoProcessor,
    VisionEncoderDecoderConfig,
)
from PIL import Image
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

from be.config.settings import settings
from be.service import model_registry
from be.utils.logger import get_logger

logger = get_logger(__name__)

# DEVICE
if settings.DATABRICKS_DEVICE and settings.DATABRICKS_DEVICE != "auto":
    DEVICE = torch.device(settings.DATABRICKS_DEVICE)
    logger.info(f"Using compute device: {DEVICE} (from DATABRICKS_DEVICE)")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {DEVICE} (auto-detected)")

if DEVICE.type == "cuda":
    DEVICE = torch.device(
        "cuda", DEVICE.index if DEVICE.index is not None else torch.cuda.current_device()
    )
    logger.info(f"Normalized CUDA device to: {DEVICE}")


ml_components: OrderedDict[str, dict] = OrderedDict()

_components_lock = threading.Lock()

TASK_PROMPT = "<s_cord-v2>"
WARMUP_MAX_LENGTH = 10


def _max_resident_models() -> int:
    return max(1, int(settings.MAX_RESIDENT_MODELS))


def _evict_overflow() -> None:
    while len(ml_components) > _max_resident_models():
        evicted_run_id, components = ml_components.popitem(last=False)
        logger.info(
            f"Evicted model {evicted_run_id} ({components.get('name')}) "
            f"to stay within {_max_resident_models()} resident model(s)."
        )
        components.pop("model", None)
        torch.cuda.empty_cache()


def _build_quantized_model(run_id: str, paths) -> torch.nn.Module:
    """
    Rebuilds VisionEncoderDecoder model architecture, applies INT8 dynamic quantization
    to the decoder, and loads the exported INT8 state_dict from a .pt file.

    Args:
        run_id (str): MLflow run ID of the model to load
        paths (ModelPaths): Object containing artifact and model dir paths
    """

    pt_files = [f for f in os.listdir(paths.artifact_dir) if f.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found for quantized model {run_id}")
    state_dict_path = os.path.join(paths.artifact_dir, pt_files[0])

    config = VisionEncoderDecoderConfig.from_pretrained(paths.artifact_dir)
    model = VisionEncoderDecoderModel(config)

    model.to(DEVICE)
    model.eval()
    quantize_(model.decoder, Int8DynamicActivationInt8WeightConfig())

    state_dict = torch.load(
        state_dict_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(state_dict, assign=True)

    return model


def _load_single_model(run_id: str, paths) -> dict:
    """
    Loads individual model components (Processor and VisionEncoderDecoderModel)
    from local artifacts, supporting both standard Hugging Face and INT8 quantized formats.

    Args:
        run_id (str): Mlflow run ID of the model
        paths (ModelPaths): Object storing processor, model, and artifact paths
    
    Returns:
        dict: Contain loaded components with structure:
            {
                "model": VisionEncoderDecoderModel,
                "processor": AutoProcessor,
                "is_quantized": bool,
                "name": str
            }
    """

    metadata = model_registry.read_metadata(paths, run_id)

    is_quantized = bool(metadata.get("is_quantized", False)) if metadata else False
    model_name = metadata.get("model_name", run_id) if metadata else run_id

    logger.info(f"Loading model with run_id {run_id} ({model_name})...")

    processor = AutoProcessor.from_pretrained(paths.processor_dir)

    if is_quantized:
        logger.info(f"Loading run_id {run_id} as quantized model on {DEVICE}...")
        model = _build_quantized_model(run_id, paths)

        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.eos_token_id = processor.tokenizer.eos_token_id
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
            TASK_PROMPT
        )
        model.config.vocab_size = model.config.decoder.vocab_size
        model.eval()
    else:
        logger.info(f"Loading run_id {run_id} Hugging Face model into memory on {DEVICE}...")
        model = VisionEncoderDecoderModel.from_pretrained(paths.model_dir)
        model.to(DEVICE)

    return {
        "model": model,
        "processor": processor,
        "is_quantized": is_quantized,
        "name": model_name,
    }


def load_ml_components():
    """
    Download artifacts from Databricks and register model metadata.

    Models are NOT built here; they are constructed lazily on first use and
    kept within MAX_RESIDENT_MODELS via LRU eviction.
    """

    registered = model_registry.download_all(settings.DATABRICKS_RUN_IDS)

    logger.info(
        f"{len(registered)} model(s) registered successfully "
        f"(lazy loading enabled, up to {_max_resident_models()} resident). Server ready."
    )


def clear_ml_components():
    """
    clear cache model
    """

    with _components_lock:
        ml_components.clear()
    torch.cuda.empty_cache()
    logger.info("ML components cleared.")


def warmup_models(target_ids: list[str] | None = None):
    """
    Executes dummy inference across registered models to pre-warm the system

    Initializes CUDA kernels, quantization routines, and caches beforehand so
    subsequent user requests do not encounter cold-start latency.

    Args:
        target_ids (list[str] | None, optional): Priority list of run IDs to warmup
                                                 Defaults to registration order if None.
    """

    known = model_registry.registered_models()
    candidates = list(target_ids) if target_ids else list(known.keys())
    targets = [rid for rid in candidates if rid in known][:_max_resident_models()]
    if not targets:
        return

    logger.info(f"Warming up {len(targets)} model(s)...")
    dummy_image = Image.new("RGB", (256, 256), color="white")

    for run_id in targets:
        started = time.perf_counter()
        try:
            components, _ = _get_components(run_id)
            model = components["model"]
            processor = components["processor"]
            pixel_values = processor(dummy_image, return_tensors="pt").pixel_values.to(DEVICE)
            decoder_input_ids = processor.tokenizer(
                TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(DEVICE)
            with torch.inference_mode():
                model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=WARMUP_MAX_LENGTH,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                )
            elapsed = time.perf_counter() - started
            logger.info(f"Warmup complete for {run_id} in {elapsed:.2f}s")
        except Exception as exc:
            logger.warning(f"Warmup failed for {run_id} (skipped): {exc}")


def _resolve_component_key(model_key: str) -> str:
    """
    Map a request identifier to a configured model key. Accepts the run id
    directly or a display name from metadata/registration.
    """

    with _components_lock:
        if model_key in ml_components:
            return model_key

    registered = model_registry.registered_models()

    if model_key in registered:
        return model_key

    for run_id, info in registered.items():
        if info.get("name") == model_key:
            logger.info(f"Resolved display name '{model_key}' to run_id '{run_id}'.")
            return run_id

    raise ValueError(f"Model {model_key} is not loaded or does not exist.")


def _get_components(model_key: str) -> tuple[dict, float]:
    """
    Retrieves model components from memory cache.
    If the model is not resident, performs thread-safe lazy-loading into the targeted device
    updates LRU (Least Recently Used) ordering, and evicts overflow models if limits are exceeded

    Args:
        model_key (str): Model identifier (run_id or registered display name)
    
    Returns:
        tuple[dict, float]: Tuple containing (components_dict, load_duration_seconds)
    """

    with _components_lock:
        if model_key in ml_components:
            ml_components.move_to_end(model_key)
            return ml_components[model_key], 0.0

        logger.info(f"Lazy-loading model '{model_key}' onto {DEVICE}...")
        started = time.perf_counter()

        components = _load_single_model(model_key, model_registry.get_paths(model_key))
        load_s = time.perf_counter() - started

        ml_components[model_key] = components
        ml_components.move_to_end(model_key)
        _evict_overflow()

        logger.info(
            f"Model '{model_key}' ready in {load_s:.2f}s "
            f"({len(ml_components)} resident)."
        )
        return components, load_s


def run_extraction(
    model_name: str | None = None,
    file_bytes: bytes | None = None,
    *,
    model_id: str | None = None,
) -> dict:
    """
    Executes document information extraction (OCR/Parsing) on an input image using DONUT.

    Handles image loading, task prompt initialization (`<s_cord-v2>`), model generation,
    special token cleanup, and final JSON payload conversion.

    Args:
        model_name (str | None, optional): Display name of the model
        file_bytes (bytes | None, optional): Raw bytes of the uploaded image file
        model_id (str | None, optional): Mlflow run ID of the model
    
    Returns:
        dict: Extraction prediction payload and timing metrics structured as:
            {
                "prediction": dict | str, # Parsed JSON | raw string fallback
                "timing": {
                    "load_s": float, # Model load time
                    "inference_s": float # pure inference time
                }
            }
    """

    model_key = model_id or model_name
    if not model_key:
        raise ValueError("No model specified for extraction.")

    logger.info(f"Running extraction using model: {model_key}")

    components, load_s = _get_components(_resolve_component_key(model_key))
    processor = components["processor"]
    model = components["model"]

    device = DEVICE

    inference_started = time.perf_counter()
    
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid or unreadable image file.") from exc

    with torch.inference_mode():
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        # donut requires a task prompt for generation
        decoder_input_ids = processor.tokenizer(
            TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)

        # generate output
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.config.decoder.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # decode output
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
            processor.tokenizer.pad_token, ""
        )
        sequence = re.sub(
            r"<.*?>", "", sequence, count=1
        ).strip()

        try:
            prediction = processor.token2json(sequence)
        except Exception:
            prediction = sequence

    inference_s = time.perf_counter() - inference_started

    return {
        "prediction": prediction,
        "timing": {
            "load_s": round(load_s, 3),
            "inference_s": round(inference_s, 3),
        },
    }


def preload_model(
    model_name: str | None = None,
    *,
    model_id: str | None = None,
) -> dict:
    """
    Resolves and loads a model into memory without executing inference

    Ensures that a subsequent `run_extraction` call on the model avoids
    one-time loading tlatencies.

    Args:
        model_name (str | None, optional): Display name of the model
        model_id (str | None, optional): Mlflow run ID of the model

    Returns:
        dict: Preload execution status:
            {
                "id": str,
                "name": str,
                "is_quantized": bool,
                "was_loaded": bool, # check if model was already cached
                "load_s": float # duration on load operation in seconds
            }
    """

    model_key = model_id or model_name
    if not model_key:
        raise ValueError("No model specified for preloading.")

    resolved = _resolve_component_key(model_key)
    components, load_s = _get_components(resolved)

    return {
        "id": resolved,
        "name": components.get("name", resolved),
        "is_quantized": bool(components.get("is_quantized", False)),
        "was_loaded": load_s == 0.0,
        "load_s": round(load_s, 3),
    }
