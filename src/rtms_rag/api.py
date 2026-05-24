import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .rag import generate_answer, generate_answer_with_reflexion
from .yandex_embed import get_embedding

app = FastAPI()
RETURN_QUERY_EMBEDDING = os.getenv("RETURN_QUERY_EMBEDDING", "0").strip() == "1"

class Query(BaseModel):
    question: str
    case_no: Optional[str] = None
    file_name: Optional[str] = None
    use_reflexion: bool = False
    max_iters: int = 3
    task_type: str = "qa_generation"

@app.post("/ask")
def ask(q: Query):
    try:
        trace = None
        if q.use_reflexion:
            answer_text, sources, trace = generate_answer_with_reflexion(
                q.question,
                top_k=5,
                case_no=q.case_no,
                file_name=q.file_name,
                max_iters=q.max_iters,
                task_type=q.task_type,
            )
        else:
            answer_text, sources = generate_answer(
                q.question,
                top_k=5,
                case_no=q.case_no,
                file_name=q.file_name,
                task_type=q.task_type,
            )

        response = {"answer": answer_text, "sources": sources}
        if RETURN_QUERY_EMBEDDING:
            response["embedding"] = get_embedding(q.question)
        if trace is not None:
            response["reflexion_trace"] = trace
        return response

    except Exception as e:
        return {"error": str(e)}
