from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    """
    System health check endpoint.
    """
    return {"status": "ok", "message": "System is healthy"}
