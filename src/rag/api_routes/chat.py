from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/chat", tags=["chat"])


class CreateSession(BaseModel):
    title: str = "New Chat"
    collection_id: str | None = None


@router.get("/sessions")
def list_sessions():
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    sessions = pg.list_chat_sessions()
    result = []
    for s in sessions:
        row = dict(s)
        for k, v in row.items():
            if hasattr(v, 'isoformat'):
                row[k] = v.isoformat()
            elif hasattr(v, '__str__') and not isinstance(v, (str, int, float, bool, type(None), list, dict)):
                row[k] = str(v)
        result.append(row)
    return {"sessions": result}


@router.post("/sessions")
def create_session(body: CreateSession):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    session = pg.create_chat_session(body.title, body.collection_id)
    return {"id": str(session["id"]), "title": session["title"]}


@router.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    messages = pg.get_chat_messages(session_id)
    result = []
    for m in messages:
        row = dict(m)
        for k, v in row.items():
            if hasattr(v, 'isoformat'):
                row[k] = v.isoformat()
            elif hasattr(v, '__str__') and not isinstance(v, (str, int, float, bool, type(None), list, dict)):
                row[k] = str(v)
        result.append(row)
    return {"messages": result}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    deleted = pg.delete_chat_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}
