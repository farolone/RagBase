from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["ratings"])


class FeedbackRequest(BaseModel):
    session_id: str | None = None
    question: str
    answer: str
    rating: int  # -1 or +1
    comment: str = ""


@router.post("/feedback")
def submit_feedback(body: FeedbackRequest):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    pg.save_feedback(body.session_id, body.question, body.answer, body.rating, body.comment)
    return {"ok": True}
