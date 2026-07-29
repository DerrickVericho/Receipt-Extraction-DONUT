import uuid
from fastapi import UploadFile, HTTPException
from service.databricks_service import get_available_models, run_extraction
from schema.schemas import BaseResponse
from utils.logger import get_logger, new_request_dir, _save_uploaded_file, _save_log

logger = get_logger(__name__)


def get_models_flow() -> BaseResponse:
    logger.info("Starting get_models flow")

    try:
        models = get_available_models()
        return BaseResponse(
            success=True, msg="Successfully retrieved models", data=models
        )

    except Exception as e:
        logger.error(f"Error in get_models_flow: {str(e)}")
        raise HTTPException(500, f"Failed to retrieve models: {str(e)}")


def extract_flow(model_name: str, file: UploadFile) -> BaseResponse:

    # making logs directory
    request_id = str(uuid.uuid4())
    logger.info(
        f"Starting extract flow for model: {model_name}, request_id: {request_id}"
    )
    request_dir = new_request_dir(request_id)

    try:
        # save uploaded file
        file_bytes, original_filename = _save_uploaded_file(request_dir, file)

        # run extraction model
        extracted_data = run_extraction(model_name, file_bytes)

        # save success log
        _save_log(
            request_dir,
            {
                "request_id": request_id,
                "model_name": model_name,
                "image_file": original_filename,
                "status": "success",
                "result": extracted_data,
            },
        )

        return BaseResponse(
            success=True, msg="Extraction successful", data=extracted_data
        )

    except Exception as e:
        logger.error(f"Error in extract_flow [{request_id}]: {str(e)}")

        # save error log
        _save_log(
            request_dir,
            {
                "request_id": request_id,
                "model_name": model_name,
                "status": "error",
                "error_message": str(e),
            },
        )

        raise HTTPException(500, f"Extraction failed: {str(e)}")
