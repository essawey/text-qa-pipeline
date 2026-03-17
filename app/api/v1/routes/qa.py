from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
import os
import uuid
from openai import OpenAI
from sqlalchemy.orm import Session

# Import your database session and models
from app.db.database import get_db
from app.db.models import Query, Response
from dotenv import load_dotenv
load_dotenv(override=True)
router = APIRouter()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class QuestionRequest(BaseModel):
    query: str
    prepare_tts: bool = False

@router.post("/ask-question")
def ask_question(
    request_body: QuestionRequest, 
    request: Request, 
    db: Session = Depends(get_db)  # Inject DB session here
):
    """
    Main Q&A endpoint. 
    Can optionally prepare TTS (Text-To-Speech) for the answer.
    Saves the query and response to the PostgreSQL database.
    """
    try:
        rag_pipeline = request.app.state.rag_pipeline
        qa_pipeline = request.app.state.qa_pipeline
        
        search_results = rag_pipeline.search(request_body.query)
        answer_result = qa_pipeline.generate_answer(request_body.query, search_results)
        
        # generation.py returns a dictionary with 'answer' and 'confidence'
        answer_text = answer_result.get("answer", "")
        
        # --- NEW: Save to Database ---
        # 1. Save the Query
        db_query = Query(text=request_body.query)
        db.add(db_query)
        db.commit()
        db.refresh(db_query)
        
        # 2. Extract basic source info (optional, keeps JSON clean)
        sources_data = [
            {"chunk_id": res["chunk_id"], "score": res["score"]} 
            for res in search_results
        ]
        
        # 3. Save the Response
        db_response = Response(
            query_id=db_query.id,
            text=answer_text,
            sources=sources_data
        )
        db.add(db_response)
        db.commit()
        # -----------------------------

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    audio_url = None
    if request_body.prepare_tts:
        try:
            audio_dir = os.path.join(os.getcwd(), "static", "audio")
            os.makedirs(audio_dir, exist_ok=True)
            filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
            file_path = os.path.join(audio_dir, filename)
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",      
                input=answer_text
            )
            response.stream_to_file(file_path)
            audio_url = f"/static/audio/{filename}"
        except Exception as e:
            print(f"Failed to generate TTS from OpenAI: {e}")
    
    return {
        "status": "success",
        "query": request_body.query,
        "answer": answer_result,
        "tts_prepared": request_body.prepare_tts and audio_url is not None,
        "tts_audio_url": audio_url
    }