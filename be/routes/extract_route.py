from fastapi import APIRouter, UploadFile, File, Form
from controller.extract_controller import get_models_flow, extract_flow
from schema.schemas import BaseResponse

router = APIRouter()

@router.get("/models", response_model=BaseResponse)
def get_models():
    """
    Get models for the model selection in streamlit
    """
    return get_models_flow()

@router.post("/extract", response_model=BaseResponse)
def extract_information(model_name: str = Form(...), file: UploadFile = File(...)):
    """
    Extract information from the uploaded file using the selected model
    """
    return extract_flow(model_name, file)
