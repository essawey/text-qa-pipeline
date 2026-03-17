from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from app.api.v1.routes import api_router

app = FastAPI()

from src.indexing import DocumentIndexingPipeline
from src.evaluation import PipelineEvaluator
from src.data_processing import process_and_chunk_data
import src.generation as generation
import pandas as pd

from app.db.database import engine
from app.db import models

# Create all tables
models.Base.metadata.create_all(bind=engine)

@app.on_event("startup")
def startup_event():
    rag_pipeline = DocumentIndexingPipeline()
    
    base_csv_path = os.path.join(os.path.dirname(__file__), "..", "Natural-Questions-Base.csv")
    df_base = pd.read_csv(base_csv_path, nrows=100).dropna()
    chunks_df, _ = process_and_chunk_data(df_base)
    rag_pipeline.add_documents(chunks_df)

    app.state.rag_pipeline = rag_pipeline
    app.state.qa_pipeline = generation
    app.state.evaluator = PipelineEvaluator()

app.include_router(api_router, prefix="/api/v1")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
