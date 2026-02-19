import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/sources", tags=["sources"])


class CreateSourceConfig(BaseModel):
    source_type: str
    name: str
    config: dict = {}
    collection_id: str | None = None
    enabled: bool = True


class UpdateSourceConfig(BaseModel):
    source_type: str | None = None
    name: str | None = None
    config: dict | None = None
    collection_id: str | None = None
    enabled: bool | None = None


def _serialize_row(row):
    result = dict(row)
    for k, v in result.items():
        if hasattr(v, "isoformat"):
            result[k] = v.isoformat()
        elif hasattr(v, "__str__") and not isinstance(v, (str, int, float, bool, type(None), list, dict)):
            result[k] = str(v)
    return result


@router.get("")
def list_sources(source_type: str | None = None):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    rows = pg.list_source_configs(source_type=source_type)
    return {"sources": [_serialize_row(r) for r in rows]}


@router.post("")
def create_source(body: CreateSourceConfig):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    row = pg.create_source_config(
        source_type=body.source_type,
        name=body.name,
        config=body.config,
        collection_id=body.collection_id,
        enabled=body.enabled,
    )
    return _serialize_row(row)


@router.get("/{config_id}")
def get_source(config_id: str):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    row = pg.get_source_config(config_id)
    if not row:
        raise HTTPException(status_code=404, detail="Source config not found")
    return _serialize_row(row)


@router.put("/{config_id}")
def update_source(config_id: str, body: UpdateSourceConfig):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    row = pg.update_source_config(config_id, **fields)
    if not row:
        raise HTTPException(status_code=404, detail="Source config not found")
    return _serialize_row(row)


@router.delete("/{config_id}")
def delete_source(config_id: str):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    deleted = pg.delete_source_config(config_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Source config not found")
    return {"deleted": True}


@router.post("/{config_id}/toggle")
def toggle_source(config_id: str):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    row = pg.get_source_config(config_id)
    if not row:
        raise HTTPException(status_code=404, detail="Source config not found")
    updated = pg.update_source_config(config_id, enabled=not row["enabled"])
    return _serialize_row(updated)


@router.post("/import-yaml")
def import_yaml():
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    count = pg.import_sources_from_yaml("/root/rag/sources.yaml")
    return {"imported": count}
