import logging
import os
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler

from fastapi import UploadFile

# semua log hidup di be/logs/ -> app.log + requests/<tanggal>/<request_id>/
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
REQUEST_LOGS_DIR = os.path.join(LOGS_DIR, "requests")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        # ponytail: rotasi 5MB x 3 backup, ganti TimedRotatingFileHandler kalau butuh rotasi harian
        RotatingFileHandler(
            os.path.join(LOGS_DIR, "app.log"),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

# uvicorn punya logger sendiri (propagate=False), tanpa ini request & error startup
# tidak pernah masuk app.log
for _name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(_name).handlers = logging.getLogger().handlers


def get_logger(name: str):
    return logging.getLogger(name)


def new_request_dir(request_id: str) -> str:
    """
    create the artifact folder for one request: logs/requests/YYYY-MM-DD/<request_id>/
    """

    path = os.path.join(
        REQUEST_LOGS_DIR, datetime.now().strftime("%Y-%m-%d"), request_id
    )
    os.makedirs(path, exist_ok=True)
    return path


def _save_uploaded_file(request_dir: str, file: UploadFile) -> tuple[bytes, str]:
    """
    helper to read and save the uploaded file to local directory
    """

    # basename: filename datang dari client, tanpa ini "../../x.jpg" nulis di luar request_dir
    file_bytes = file.file.read()
    original_filename = os.path.basename(file.filename or "") or "image.jpg"
    with open(os.path.join(request_dir, original_filename), "wb") as f:
        f.write(file_bytes)

    return file_bytes, original_filename


def _save_log(request_dir: str, log_data: dict):
    """
    helper to write result or error logs to JSON.
    """

    # save log file
    with open(
        os.path.join(request_dir, "result.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)
