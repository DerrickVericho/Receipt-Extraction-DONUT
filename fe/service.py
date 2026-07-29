import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL")


def fetch_models():
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("data"):
                return data["data"]
    except Exception:
        pass
    return None


def extract_receipt(selected_model, file_bytes, file_name, file_type):
    data = {"model_name": selected_model}
    files = {"file": (file_name, file_bytes, file_type)}
    return requests.post(f"{API_BASE_URL}/extract", files=files, data=data)
