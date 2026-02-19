from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/collections", tags=["collections"])


class CreateCollection(BaseModel):
    name: str
    description: str = ""
    color: str = "#3B82F6"


class UpdateCollection(BaseModel):
    name: str
    description: str = ""
    color: str = "#3B82F6"


class AddDocument(BaseModel):
    document_id: str


@router.get("")
def list_collections():
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    collections = pg.list_collections()
    result = []
    for c in collections:
        row = dict(c)
        for k, v in row.items():
            if hasattr(v, 'isoformat'):
                row[k] = v.isoformat()
            elif hasattr(v, '__str__') and not isinstance(v, (str, int, float, bool, type(None), list, dict)):
                row[k] = str(v)
        result.append(row)
    return {"collections": result}


@router.post("")
def create_collection(body: CreateCollection):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    coll = pg.create_collection(body.name, body.description, body.color)
    return {"id": str(coll["id"]), "name": coll["name"]}


@router.get("/{collection_id}")
def get_collection(collection_id: str):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    coll = pg.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")
    row = dict(coll)
    for k, v in row.items():
        if hasattr(v, 'isoformat'):
            row[k] = v.isoformat()
        elif hasattr(v, '__str__') and not isinstance(v, (str, int, float, bool, type(None), list, dict)):
            row[k] = str(v)
    return row


@router.put("/{collection_id}")
def update_collection(collection_id: str, body: UpdateCollection):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    updated = pg.update_collection(collection_id, body.name, body.description, body.color)
    if not updated:
        raise HTTPException(status_code=404, detail="Collection not found")
    return {"ok": True}


@router.delete("/{collection_id}")
def delete_collection(collection_id: str):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    deleted = pg.delete_collection(collection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Collection not found")
    return {"deleted": True}


@router.post("/{collection_id}/documents")
def add_document_to_collection(collection_id: str, body: AddDocument):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    pg.add_document_to_collection(body.document_id, collection_id)
    return {"ok": True}


@router.delete("/{collection_id}/documents/{document_id}")
def remove_document_from_collection(collection_id: str, document_id: str):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    pg.remove_document_from_collection(document_id, collection_id)
    return {"ok": True}
