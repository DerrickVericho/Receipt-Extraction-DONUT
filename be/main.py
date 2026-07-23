from fastapi import FastAPI
from contextlib import asynccontextmanager
from routes.extract_route import router as extract_router
from utils.logger import get_logger
from service.databricks_service import load_ml_components, clear_ml_components
import uvicorn

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application startup: initializing models")
    load_ml_components()
    yield
    logger.info("FastAPI application shutdown: clearing models")
    clear_ml_components()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Databricks OCR API", lifespan=lifespan)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, modify this in production to specific frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

app.include_router(extract_router, prefix="/api", tags=["Extraction"])

@app.get("/")
def root():
    return {"message": "Welcome to Databricks OCR API"}