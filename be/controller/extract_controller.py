import uuid
import os
import json
from fastapi import UploadFile
from service.databricks_service import get_available_models, run_extraction
from utils.helpers import format_response
from utils.logger import get_logger, _save_uploaded_file, _save_log

logger = get_logger(__name__)

REQUEST_LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "requests")
os.makedirs(REQUEST_LOGS_DIR, exist_ok=True)

def get_models_flow():
    logger.info("Starting get_models flow")

    try:
        models = get_available_models()
        return format_response(success=True, msg="Successfully retrieved models", data=models)
        
    except Exception as e:
        logger.error(f"Error in get_models_flow: {str(e)}")
        return format_response(success=False, msg=f"Failed to retrieve models: {str(e)}")

def extract_flow(model_name: str, file: UploadFile):

    # making logs directory
    request_id = str(uuid.uuid4())
    logger.info(f"Starting extract flow for model: {model_name}, request_id: {request_id}")
    request_dir = os.path.join(REQUEST_LOGS_DIR, request_id)
    os.makedirs(request_dir, exist_ok=True)
    
    try:
        # save uploaded file
        file_bytes, original_filename = _save_uploaded_file(request_dir, file)
        
        # run extraction model
        extracted_data = run_extraction(model_name, file_bytes)
        
        # save success log
        _save_log(request_dir, {
            "request_id": request_id,
            "model_name": model_name,
            "image_file": original_filename,
            "status": "success",
            "result": extracted_data
        })
            
        return format_response(success=True, msg="Extraction successful", data=extracted_data)
        
    except Exception as e:
        logger.error(f"Error in extract_flow [{request_id}]: {str(e)}")
        
        # save error log
        _save_log(request_dir, {
            "request_id": request_id,
            "model_name": model_name,
            "status": "error",
            "error_message": str(e)
        })
            
        return format_response(success=False, msg=f"Extraction failed: {str(e)}")
