from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel
import pandas as pd
import os
from src.data_processing import process_and_chunk_data

router = APIRouter()

class EvaluationRequest(BaseModel):
    sample_size: int = 10

def run_evaluation_task(evaluator, rag_pipeline, qa_pipeline, sample_size: int):
    base_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Natural-Questions-Base.csv"))
    df_base = pd.read_csv(base_csv_path, nrows=100).dropna()
    _, df_base_processed = process_and_chunk_data(df_base)
    
    test_df = df_base_processed.sample(min(sample_size, len(df_base_processed)), random_state=2026)
    
    for index, row in test_df.iterrows():
        query_text = row['question']
        expected_source = index
        reference_answer = row['long_answers_clean']

        evaluator.track_query(
            query_id=index,
            query_text=query_text,
            expected_source_id=expected_source,
            reference_answer=reference_answer,
            search_fn=lambda q: rag_pipeline.search(q, k=1),
            generate_fn=lambda q, ctx: qa_pipeline.generate_answer(q, ctx)
        )
    evaluator.generate_report()

@router.post("/evaluate")
def evaluate_model(request: Request, body: EvaluationRequest, background_tasks: BackgroundTasks):

    evaluator = request.app.state.evaluator
    rag_pipeline = request.app.state.rag_pipeline
    qa_pipeline = request.app.state.qa_pipeline
    
    background_tasks.add_task(
        run_evaluation_task, 
        evaluator, 
        rag_pipeline, 
        qa_pipeline, 
        body.sample_size
    )
    
    return {
        "status": "evaluation_started",
        "sample_size": body.sample_size,
        "message": "Evaluation process has been queued."
    }
