from fastapi import FastAPI
from contextlib import asynccontextmanager
from be.routes.extract_route import router as extract_router
from be.utils.logger import get_logger
from be.service.donut_service import load_ml_components, clear_ml_components, warmup_models
from be.service.model_catalog import load_experiment_results, get_available_models

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application startup: initializing models")
    load_ml_components()
    load_experiment_results()
    catalog = get_available_models()
    warmup_models(
        [m["id"] for m in sorted(catalog, key=lambda m: not m["recommended"])]
    )
    yield
    logger.info("FastAPI application shutdown: clearing models")
    clear_ml_components()


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Databricks OCR API", lifespan=lifespan)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allows all origins, modify this in production to specific frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

app.include_router(extract_router, prefix="/api", tags=["Extraction"])


@app.get("/")
def root():
    return {"message": "Welcome to Databricks OCR API"}
