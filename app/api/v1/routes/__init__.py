from fastapi import APIRouter
from . import health, qa, evaluation, db_view

api_router = APIRouter()

# Include all individual route modules
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(qa.router, tags=["Q&A"])
api_router.include_router(evaluation.router, tags=["Evaluation"])
api_router.include_router(db_view.router, tags=["Database"], prefix="/db")
