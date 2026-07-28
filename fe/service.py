import requests
import os

from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL")


def fetch_models():
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("data"):
                return {item["id"]: item["description"] for item in data["data"]}
    except Exception as e:
        pass
    return None


def extract_receipt(uploaded_file, selected_model):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    data = {"model_name": selected_model}
    return requests.post(f"{API_BASE_URL}/extract", files=files, data=data)
