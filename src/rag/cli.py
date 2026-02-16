import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="rag", help="RAG Wissensmanagement System")
console = Console()


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Path or URL to ingest"),
    type: str = typer.Option("auto", help="Source type: pdf, youtube, web, reddit, twitter, auto"),
):
    """Ingest a single document."""
    from rag.ingestion.pdf import PDFIngestor
    from rag.ingestion.youtube import YouTubeIngestor
    from rag.ingestion.web import WebIngestor
    from rag.processing.embedding import Embedder
    from rag.processing.ner import EntityExtractor
    from rag.processing.graph_builder import GraphBuilder
    from rag.storage.qdrant import QdrantStore
    from rag.storage.postgres import PostgresStore

    # Auto-detect type
    if type == "auto":
        if source.endswith(".pdf"):
            type = "pdf"
        elif "youtube.com" in source or "youtu.be" in source:
            type = "youtube"
        elif "reddit.com" in source:
            type = "reddit"
        elif "twitter.com" in source or "x.com" in source:
            type = "twitter"
        else:
            type = "web"

    console.print(f"[bold]Ingesting[/bold] {source} as [cyan]{type}[/cyan]")

    ingestors = {
        "pdf": PDFIngestor,
        "youtube": YouTubeIngestor,
        "web": WebIngestor,
    }

    if type not in ingestors:
        console.print(f"[red]Unsupported type: {type}[/red]")
        raise typer.Exit(1)

    ingestor = ingestors[type]()
    doc, chunks = ingestor.ingest(source)
    console.print(f"  Extracted [green]{len(chunks)}[/green] chunks")

    # Embed and store
    embedder = Embedder()
    qdrant = QdrantStore()
    qdrant.ensure_collection()
    postgres = PostgresStore()

    for chunk in chunks:
        emb = embedder.embed(chunk.content)
        qdrant.upsert(
            chunk=chunk,
            dense_vector=emb.dense,
            sparse_indices=emb.sparse_indices,
            sparse_values=emb.sparse_values,
        )

    postgres.save_document(doc)

    # NER + Graph
    ner = EntityExtractor()
    graph = GraphBuilder()
    all_entities = []
    for chunk in chunks:
        entities = ner.extract(chunk.content, document_id=doc.id, chunk_id=chunk.id)
        all_entities.extend(entities)

    graph.process_document(doc, all_entities)
    graph.close()

    console.print(f"  Found [green]{len(all_entities)}[/green] entities")
    console.print(f"[bold green]Done![/bold green] Document ID: {doc.id}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    platform: str | None = typer.Option(None, help="Filter by platform"),
    limit: int = typer.Option(10, help="Number of results"),
):
    """Search documents with hybrid retrieval."""
    from rag.retrieval.hybrid import HybridRetriever

    console.print(f"[bold]Searching:[/bold] {query}")
    retriever = HybridRetriever()
    results = retriever.retrieve(query, limit=limit, filter_platform=platform)

    table = Table(title=f"Results ({len(results)})")
    table.add_column("Score", style="cyan", width=8)
    table.add_column("Content", width=60)
    table.add_column("Platform", style="green", width=10)

    for r in results:
        table.add_row(
            f"{r.score:.3f}",
            r.content[:100] + "..." if len(r.content) > 100 else r.content,
            r.metadata.get("platform", "?"),
        )
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer"),
    platform: str | None = typer.Option(None, help="Filter by platform"),
):
    """Ask a question with cited answers."""
    from rag.retrieval.hybrid import HybridRetriever
    from rag.generation.router import QueryRouter
    from rag.generation.citation import CitationGenerator

    console.print(f"[bold]Question:[/bold] {question}")

    retriever = HybridRetriever()
    results = retriever.retrieve(question, limit=10, filter_platform=platform)

    if not results:
        console.print("[yellow]No relevant documents found.[/yellow]")
        raise typer.Exit(0)

    citation_gen = CitationGenerator()
    prompt, citation_map = citation_gen.build_prompt(question, results)

    router = QueryRouter()
    answer = router.generate(question, context=prompt, system=citation_gen.SYSTEM_PROMPT)

    parsed = citation_gen.parse_citations(answer, citation_map)

    console.print(f"\n[bold]Answer:[/bold]\n{parsed['answer']}\n")
    if parsed["sources"]:
        console.print("[bold]Sources:[/bold]")
        for s in parsed["sources"]:
            console.print(f"  [{s['ref']}] {s.get('source_url', s['document_id'])}")


@app.command()
def stats():
    """Show system statistics."""
    from rag.storage.qdrant import QdrantStore
    from rag.storage.postgres import PostgresStore

    qdrant = QdrantStore()
    postgres = PostgresStore()

    try:
        info = qdrant.get_collection_info()
        console.print(f"[bold]Qdrant:[/bold] {info.points_count} vectors in '{qdrant.collection_name}'")
    except Exception:
        console.print("[yellow]Qdrant: collection not initialized[/yellow]")

    docs = postgres.search_documents(limit=1000)
    console.print(f"[bold]PostgreSQL:[/bold] {len(docs)} documents")

    from collections import Counter
    platforms = Counter(d.platform.value for d in docs)
    for p, count in platforms.most_common():
        console.print(f"  {p}: {count}")


if __name__ == "__main__":
    app()
