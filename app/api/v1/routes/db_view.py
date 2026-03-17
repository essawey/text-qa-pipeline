from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Query, Response

router = APIRouter()

@router.get("/queries")
def get_queries(db: Session = Depends(get_db)):
    queries = db.query(Query).all()
    return queries

@router.get("/responses")
def get_responses(db: Session = Depends(get_db)):
    responses = db.query(Response).all()
    return responses
