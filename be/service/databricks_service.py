import os
import io
import json
import re

import mlflow
import torch
from mlflow.tracking import MlflowClient
from transformers import VisionEncoderDecoderModel, AutoProcessor
from PIL import Image

from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

# config from reference
ARTIFACT_PATH = "model"
LOCAL_DOWNLOAD_DIR = "./model_cache"

# determine device
if settings.DATABRICKS_DEVICE and settings.DATABRICKS_DEVICE != "auto":
    DEVICE = torch.device(settings.DATABRICKS_DEVICE)
    logger.info(f"Using compute device: {DEVICE} (from DATABRICKS_DEVICE)")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {DEVICE} (auto-detected)")

# global variable agar tidak di load berulang ulang
ml_components = {}


def load_ml_components():
    """
    load ML model from Databricks to local server. Skip download if the model is already available.
    """

    # set environment variables for Databricks authentication so mlflow can use them
    os.environ["DATABRICKS_HOST"] = settings.DATABRICKS_HOST
    os.environ["DATABRICKS_TOKEN"] = settings.DATABRICKS_TOKEN

    # point MLflow to Databricks
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    logger.info("Downloading artifacts from Databricks...")

    # download and load models from Databricks
    for run_id in settings.DATABRICKS_RUN_IDS:

        run_id = run_id.strip()
        if not run_id:
            continue
        logger.info(f"Loading model with run_id {run_id}...")

        # determine paths, using run_id as cache folder name
        model_download_dir = os.path.join(LOCAL_DOWNLOAD_DIR, run_id)
        expected_local_artifact_path = os.path.join(model_download_dir, ARTIFACT_PATH)
        metadata_path = os.path.join(expected_local_artifact_path, "metadata.json")
        processor_dir = os.path.join(expected_local_artifact_path, "processor")
        model_dir = os.path.join(expected_local_artifact_path, "huggingface_model")

        # download if not already cached (metadata.json present for both quantized and non-quantized)
        if not os.path.exists(metadata_path):
            logger.info(f"Downloading run_id {run_id} from Databricks...")
            client.download_artifacts(
                run_id=run_id, path=ARTIFACT_PATH, dst_path=model_download_dir
            )
        else:
            logger.info(f"run_id {run_id} already exists locally. Skipping download.")

        # determine model type and name from metadata.json
        model_name = run_id
        is_quantized = False
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                is_quantized = metadata.get("is_quantized", False)
                model_name = metadata.get("model_name", run_id)
            except Exception:
                pass

        # load processor (same for both types)
        processor = AutoProcessor.from_pretrained(processor_dir)

        # load model based on quantization type
        if is_quantized:
            logger.info(f"Loading run_id {run_id} as quantized model on CPU...")
            pt_files = [f for f in os.listdir(expected_local_artifact_path) if f.endswith(".pt")]
            if not pt_files:
                raise FileNotFoundError(f"No .pt file found for quantized model {run_id}")
            model_path = os.path.join(expected_local_artifact_path, pt_files[0])
            model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
            model.config.pad_token_id = processor.tokenizer.pad_token_id
            model.config.eos_token_id = processor.tokenizer.eos_token_id
            model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_cord-v2>")
            model.config.vocab_size = model.config.decoder.vocab_size
            model.eval()
        else:
            logger.info(f"Loading run_id {run_id} Hugging Face model into memory on {DEVICE}...")
            model = VisionEncoderDecoderModel.from_pretrained(model_dir)
            model.to(DEVICE)

        # set global variable
        ml_components[run_id] = {
            "model": model,
            "processor": processor,
            "is_quantized": is_quantized,
            "name": model_name,
        }

    logger.info("All configured models loaded successfully. Server ready.")


def clear_ml_components():
    """
    clear cache model
    """

    # clear the models
    ml_components.clear()
    logger.info("ML components cleared.")


def get_available_models():
    """
    get list of available models from currently loaded ml_components
    """

    # returning the models list
    logger.info("Fetching available models from Databricks")
    available_models = []
    for run_id, components in ml_components.items():
        available_models.append({
            "id": run_id,
            "description": components.get("name", run_id),
        })

    return available_models


def run_extraction(model_name: str, file_bytes: bytes):
    """
    run the extraction using the selected model
    """

    # model validation if somehow errors
    logger.info(f"Running extraction using model: {model_name}")
    if model_name not in ml_components:
        raise ValueError(f"Model {model_name} is not loaded or does not exist.")

    # variable inits
    components = ml_components[model_name]
    processor = components["processor"]
    model = components["model"]

    # quantized models must stay on CPU
    device = torch.device("cpu") if components.get("is_quantized") else DEVICE

    # prepare image for the model and move to correct device
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # donut requires a task prompt for generation
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    decoder_input_ids = decoder_input_ids.to(device)

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
    ).strip()  # remove first task start token

    # json validation
    try:
        prediction = processor.token2json(sequence)
    except Exception:
        prediction = sequence

    return {"prediction": prediction}
