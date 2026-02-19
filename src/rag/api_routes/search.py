import json
import asyncio
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    platform: str | None = Query(None),
    author: str | None = Query(None),
    collection_id: str | None = Query(None),
    limit: int = Query(10, le=100),
):
    from rag.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever()
    results = retriever.retrieve(q, limit=limit, filter_platform=platform, filter_author=author)

    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ],
    }


@router.post("/ask/stream")
async def ask_stream(request: Request):
    """SSE streaming Q&A endpoint. Streams tokens as they are generated."""
    body = await request.json()
    question = body.get("question", "")
    platform = body.get("platform")
    limit = body.get("limit", 10)
    session_id = body.get("session_id")

    if not question:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'content': 'No question provided'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def generate():
        from rag.retrieval.hybrid import HybridRetriever
        from rag.generation.citation import CitationGenerator
        from rag.generation.llm import LLMClient
        from rag.storage.postgres import PostgresStore

        retriever = HybridRetriever()
        results = retriever.retrieve(question, limit=limit, filter_platform=platform)

        if not results:
            yield f"data: {json.dumps({'type': 'content', 'content': 'No relevant documents found.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
            return

        citation_gen = CitationGenerator()
        prompt, citation_map = citation_gen.build_prompt(question, results)

        # Send sources metadata first
        sources = []
        for ref, result in citation_map.items():
            sources.append({
                "ref": ref,
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "content_preview": result.content[:300],
                "source_url": result.metadata.get("source_url", ""),
                "platform": result.metadata.get("platform", ""),
                "title": result.metadata.get("title", ""),
            })
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Stream the answer
        llm = LLMClient()
        full_answer = ""
        try:
            async for token in llm.stream_generate(
                prompt=prompt,
                system=citation_gen.SYSTEM_PROMPT,
            ):
                full_answer += token
                yield f"data: {json.dumps({'type': 'content', 'content': token})}\n\n"
        except Exception as e:
            # Fallback to non-streaming
            from rag.generation.router import QueryRouter
            router = QueryRouter()
            full_answer = router.generate(question, context=prompt, system=citation_gen.SYSTEM_PROMPT)
            yield f"data: {json.dumps({'type': 'content', 'content': full_answer})}\n\n"

        # Save to chat session if session_id provided
        if session_id:
            try:
                pg = PostgresStore()
                pg.save_chat_message(session_id, "user", question)
                pg.save_chat_message(session_id, "assistant", full_answer, sources)
            except Exception:
                pass

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
