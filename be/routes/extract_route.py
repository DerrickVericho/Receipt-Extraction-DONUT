from fastapi import APIRouter, UploadFile, File, Form
from be.controller.extract_controller import (
    get_models_flow,
    extract_flow,
    preload_model_flow,
)
from be.schema.schemas import BaseResponse

router = APIRouter()

@router.get("/models", response_model=BaseResponse)
def get_models():
    return get_models_flow()

@router.post("/models/{model_id}/preload", response_model=BaseResponse)
def preload_model(model_id: str):
    return preload_model_flow(model_id)

@router.post("/extract", response_model=BaseResponse)
def extract_information(model_name: str = Form(...), file: UploadFile = File(...)):
    return extract_flow(model_name, file)
