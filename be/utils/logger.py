import logging
import os
import json
from fastapi import UploadFile

log_file_path = os.path.join(os.path.dirname(__file__), "..", "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
)


def get_logger(name: str):
    return logging.getLogger(name)


def _save_uploaded_file(request_dir: str, file: UploadFile) -> tuple[bytes, str]:
    """
    helper to read and save the uploaded file to local directory
    """

    # save file
    file_bytes = file.file.read()
    original_filename = file.filename or "image.jpg"
    image_path = os.path.join(request_dir, original_filename)
    with open(image_path, "wb") as f:
        f.write(file_bytes)

    return file_bytes, original_filename


def _save_log(request_dir: str, log_data: dict):
    """
    helper to write result or error logs to JSON.
    """

    # save log file
    with open(os.path.join(request_dir, "result.json"), "w") as f:
        json.dump(log_data, f, indent=4)
