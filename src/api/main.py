from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config.settings import settings
from src.api.routes import upload

app = FastAPI(
    title="Outfit AI API",
    description="AI-powered outfit recommendation system",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Outfit AI API",
        "version": "0.1.0",
        "status": "running"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": "not connected yet",
        "ml_models": "not loaded yet"
    }