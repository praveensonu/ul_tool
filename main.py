from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from config.api_config import get_cors_allowed_origins
from routes.model_routes import router as model_router
from routes.dataset_routes import router as dataset_router
from routes.config_routes import router as config_router
from routes.train_routes import router as train_router
from routes.evaluation_routes import router as evaluation_router

app = FastAPI(
    title="LLM Training Control API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
)

api_router = APIRouter(prefix="/api")
api_router.include_router(model_router)
api_router.include_router(dataset_router)
api_router.include_router(config_router)
api_router.include_router(train_router)
api_router.include_router(evaluation_router)


@api_router.get("/health", tags=["Health"])
async def health():
    return {"message": "LLM training API is running"}


app.include_router(api_router)

# Preserve the original paths for existing scripts and older frontend builds.
app.include_router(model_router, include_in_schema=False)
app.include_router(dataset_router, include_in_schema=False)
app.include_router(config_router, include_in_schema=False)
app.include_router(train_router, include_in_schema=False)
app.include_router(evaluation_router, include_in_schema=False)


@app.get("/")
async def home():
    return {"message": "LLM training API is running"}


@app.get("/openapi.json", include_in_schema=False)
async def legacy_openapi():
    return JSONResponse(app.openapi())


@app.get("/docs", include_in_schema=False)
async def legacy_docs():
    return RedirectResponse(url="/api/docs")


@app.get("/redoc", include_in_schema=False)
async def legacy_redoc():
    return RedirectResponse(url="/api/redoc")
