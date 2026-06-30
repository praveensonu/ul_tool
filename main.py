from fastapi import FastAPI

from routes.model_routes import router as model_router
from routes.dataset_routes import router as dataset_router
from routes.config_routes import router as config_router
from routes.train_routes import router as train_router

app = FastAPI(title="LLM Training Control API")

app.include_router(model_router)
app.include_router(dataset_router)
app.include_router(config_router)
app.include_router(train_router)


@app.get("/")
def home():
    return {"message": "LLM training API is running"}