from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from pathlib import Path

# Import routers
from routers import attribution_router, models_router, performance_router

app = FastAPI(
    title="TokenLense API",
    description="API for accessing and visualizing model attribution data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_DIR = Path("data")  # Path to data directory


@app.get("/", response_model=Dict[str, str])
def read_root():
    """API root endpoint"""
    return {
        "name": "TokenLense API",
        "description": "API for accessing and visualizing model attribution data",
        "docs": "/docs"
    }


# Include routers
app.include_router(models_router)
app.include_router(attribution_router)
app.include_router(performance_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)