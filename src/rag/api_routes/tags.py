from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/tags", tags=["tags"])


class CreateTag(BaseModel):
    name: str
    tag_type: str = "manual"
    color: str = "#6B7280"


@router.get("")
def list_tags():
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    tags = pg.list_tags()
    result = []
    for t in tags:
        row = dict(t)
        for k, v in row.items():
            if hasattr(v, 'isoformat'):
                row[k] = v.isoformat()
            elif hasattr(v, '__str__') and not isinstance(v, (str, int, float, bool, type(None), list, dict)):
                row[k] = str(v)
        result.append(row)
    return {"tags": result}


@router.post("")
def create_tag(body: CreateTag):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    tag = pg.create_tag(body.name, body.tag_type, body.color)
    return {"id": str(tag["id"]), "name": tag["name"]}


@router.delete("/{tag_id}")
def delete_tag(tag_id: str):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    deleted = pg.delete_tag(tag_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Tag not found")
    return {"deleted": True}
