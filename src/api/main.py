from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config.settings import settings
from src.api.routes import upload
from src.database.connection import engine
from sqlalchemy import text
from src.api.routes import wardrobe, analysis

# Create FastAPI app
app = FastAPI(
    title="Outfit AI API",
    description="AI-powered outfit recommendation system",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup():
    """Check database connection on startup"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful!")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

# Include routers
app.include_router(upload.router)
app.include_router(wardrobe.router)
app.include_router(analysis.router)

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
    """Health check with database status"""
    db_status = "disconnected"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except:
        pass
    
    return {
        "status": "healthy",
        "database": db_status,
        "ml_models": "not loaded yet"
    }