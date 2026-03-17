import os
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)

def handle_fallback(reason):
    return {
        "status": "fallback",
        "answer": "I'm sorry, but I don't have enough reliable information in my current context to answer your question confidently.",
        "confidence": 0.0,
        "fallback_reason": reason
    }

def generate_answer(query, search_results, api_key=None):
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return handle_fallback("No OpenAI API key provided.")
    
    client = OpenAI(api_key=api_key)

    context_text = ""
    for result in search_results:
        context_text += (
            f"Distance Score: {result['score']:.2f}\n"
            f"Domain: {result['metadata']['question_domain']}\n"
            f"Difficulty: {result['metadata']['question_difficulty']}\n"
            f"Question Type: {result['metadata']['question_type']}\n"
            f"Source Row Index: {result['metadata']['source_row_index']}\n"
            f"Text snippet:\n{result['text']}\n"
            "---"
        )

    response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "qa_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The detailed answer based strictly on context."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "A confidence score between 0.0 and 1.0."
                    }
                },
                "required": ["answer", "confidence"],
                "additionalProperties": False
            }
        }
    }

    system_prompt = "You are a highly capable assistant. Answer using ONLY the provided context."
    user_prompt = f"Context Information:\n{context_text}\n\nUser Question: {query}"

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # Used widely instead of gpt-5-mini
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=response_schema,
        )

        response_data = json.loads(completion.choices[0].message.content)
        answer = response_data["answer"]
        confidence = response_data["confidence"]

        if confidence < 0.5:
            return handle_fallback("The context provided didn't contain enough reliable information.")

        return {
            "status": "success",
            "answer": answer,
            "confidence": confidence
        }
    except Exception as e:
        return handle_fallback(f"Error calling OpenAI: {str(e)}")
