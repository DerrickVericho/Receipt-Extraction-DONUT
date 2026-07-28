import os
import io
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using compute device: {DEVICE}")

# models available (currently static, bisa diupdate nanti)
MODELS_CONFIG = {
    "model_1": {
        "id": "model_1",
        "name": settings.DATABRICKS_MODEL_NAME_1,
        "run_id": settings.DATABRICKS_RUN_ID_1,
    },
    "model_2": {
        "id": "model_2",
        "name": settings.DATABRICKS_MODEL_NAME_2,
        "run_id": settings.DATABRICKS_RUN_ID_2,
    },
    "model_3": {
        "id": "model_3",
        "name": settings.DATABRICKS_MODEL_NAME_3,
        "run_id": settings.DATABRICKS_RUN_ID_3,
    },
}

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
    for model_key, config in MODELS_CONFIG.items():

        # getting the run id since in the mlflow, modelnya ada di run dan tidak di regist ke models
        run_id = config["run_id"]
        if not run_id:
            logger.info(f"Skipping {model_key} as no RUN_ID is provided.")
            continue
        logger.info(f"Loading {model_key} with run_id {run_id}...")

        # determine paths (using the path from databricks)
        model_download_dir = os.path.join(LOCAL_DOWNLOAD_DIR, model_key)
        expected_local_artifact_path = os.path.join(model_download_dir, ARTIFACT_PATH)
        model_dir = os.path.join(expected_local_artifact_path, "huggingface_model")
        processor_dir = os.path.join(expected_local_artifact_path, "processor")

        # check if they already exist locally
        if os.path.exists(model_dir) and os.path.exists(processor_dir):
            logger.info(f"Model {model_key} already exists locally. Skipping download.")

        # else download the model
        else:
            logger.info(f"Downloading {model_key} from Databricks...")
            local_artifact_path = client.download_artifacts(
                run_id=run_id, path=ARTIFACT_PATH, dst_path=model_download_dir
            )
            model_dir = os.path.join(local_artifact_path, "huggingface_model")
            processor_dir = os.path.join(local_artifact_path, "processor")

        # load model to local directory
        logger.info(
            f"Loading {model_key} Hugging Face model into memory on {DEVICE}..."
        )
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        model.to(DEVICE)

        # set global variable
        ml_components[model_key] = {
            "model": model,
            "processor": AutoProcessor.from_pretrained(processor_dir),
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
    get list of available models (static for now, defined in MODELS_CONFIG)
    """

    # returning the models list
    logger.info("Fetching available models from Databricks")
    available_models = []
    for model_key, config in MODELS_CONFIG.items():
        if model_key in ml_components:
            available_models.append({"id": config["id"], "description": config["name"]})

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

    # prepare image for the model and move to correct device
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

    # donut requires a task prompt for generation
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    decoder_input_ids = decoder_input_ids.to(DEVICE)

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
