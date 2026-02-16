# RAG Stack Design: Wissensmanagement-System

**Datum:** 2026-02-16
**Ziel:** Optimaler Open-Source RAG Stack fuer 50.000-100.000 Dokumente mit minimalen Verlusten bei Ingestion und Retrieval

---

## Anforderungen

| Anforderung | Detail |
|---|---|
| **Dokumente** | 50.000 - 100.000 |
| **Medientypen** | YouTube Transcripts, Twitter/X Posts, Reddit Posts, Websites, PDF Articles |
| **Sprachen** | Deutsch + Englisch (gemischt) |
| **Nutzung** | Semantische Suche + Chat/Q&A mit Quellen + Wissensexploration |
| **Tagging** | Orte, Autoren, Themen -- mit Verbindung zum Originalmedium |
| **Infrastruktur** | Self-hosted, lokal |
| **Hardware** | LXC Container auf Proxmox (192.168.178.4, x86, keine GPU) fuer DB + Pipeline + Processing. Mac Studio Ultra 512GB RAM fuer LLM + Reranker. |
| **Updates** | Taegliche Ingestion neuer Dokumente |
| **LLM aktuell** | MiniMax (wird ersetzt) |

---

## Empfohlener Stack: Uebersicht

```
                    MAC STUDIO ULTRA (512 GB, Apple Silicon)
                    ========================================
+------------------------------------------------------------------+
|                  DUAL-MODEL LLM + RERANKER (Ollama)               |
|                                                                   |
|  Qwen 2.5-72B Q8 (~75 GB)  -- Kern-RAG: Antworten + Citations   |
|  MiniMax M2.5 Q4 (~138 GB) -- Agentic Router + Code + >128K Ctx |
|  Qwen3-Reranker-8B (~10 GB)                                      |
|  Erreichbar via API: http://<mac-studio-ip>:11434                 |
+------------------------------------------------------------------+

              ^  API Calls (Ollama REST)  ^
              |                           |
    ~~~ Netzwerk (LAN 192.168.178.x) ~~~
              |                           |
              v                           v

          LXC CONTAINER auf Proxmox (192.168.178.4, x86, keine GPU)
          ==========================================================
+------------------------------------------------------------------+
|                        ANWENDUNGSSCHICHT                          |
|  CLI + FastAPI REST API                                           |
|  Chat/Q&A mit Quellen | Semantische Suche | Wissensexploration   |
+------------------------------------------------------------------+
        |                       |                       |
+------------------------------------------------------------------+
|                 RETRIEVAL (LlamaIndex)                            |
|  Hybrid Search (Dense+Sparse via Qdrant)                         |
|  -> Reranking (via Mac Studio Ultra API)                         |
|  -> LLM Generation (via Mac Studio Ultra API)                    |
+------------------------------------------------------------------+
        |                       |                       |
+-------+----------+    +------+-------+    +-----------+----------+
| VEKTOR-SUCHE     |    | KNOWLEDGE    |    | METADATEN-           |
| Qdrant (Docker)  |    | GRAPH        |    | DATENBANK            |
| Hybrid: Dense    |    | Neo4j        |    | PostgreSQL           |
|  + Sparse/BM25   |    | (Docker)     |    | (Docker)             |
+------------------+    +--------------+    +----------------------+
        ^                       ^                       ^
+------------------------------------------------------------------+
|                     VERARBEITUNGSSCHICHT (CPU)                     |
|  Embedding: BGE-M3 (via sentence-transformers, CPU)              |
|  NER: GLiNER (multilingual, zero-shot, CPU)                     |
|  Topics: BERTopic (hierarchisch, multilingual, CPU)              |
|  Chunking: Hierarchisch (Parent-Child) + Semantic                |
+------------------------------------------------------------------+
        ^
+------------------------------------------------------------------+
|                      INGESTION PIPELINE                           |
|  Orchestrierung: Prefect                                         |
|                                                                   |
|  PDF:     Docling (komplex) / PyMuPDF4LLM (einfach)             |
|  YouTube: youtube-transcript-api / yt-dlp + Whisper              |
|  Twitter: Twikit (mit Account-Pool)                              |
|  Reddit:  PRAW (OAuth, 100 req/min)                             |
|  Web:     Trafilatura (Artikel) / Crawl4AI (JS-Seiten)          |
+------------------------------------------------------------------+
```

**Hinweis CPU-Embedding:** BGE-M3 (568M Params) laeuft auf CPU akzeptabel -- langsamer als mit MPS/GPU, aber fuer taegliche Ingestion (100-500 Docs) ausreichend. Fuer den initialen Bulk-Import (100K Docs) kann das Embedding alternativ auf dem Mac Studio Ultra via API laufen.

---

## 1. Vektor-Datenbank: Qdrant

### Warum Qdrant?

| Kriterium | Qdrant | pgvector | ChromaDB | Weaviate | LanceDB |
|---|---|---|---|---|---|
| Hybrid Search (BM25+Vector) | Nativ, erstklassig | Via Extensions | Neu, unreif | Gut | Gut |
| Metadata-Filtering | Exzellent | Beste (SQL) | Schlecht (langsam) | Gut | Gut (SQL) |
| Apple Silicon | ARM64 Docker | Nativ | pip install | ARM64 Docker | Embedded |
| 100K Dokumente | Trivial | Trivial | Grenzwertig | Trivial | Trivial |
| Backup | Snapshots | pg_dump (Beste) | Datei-Kopie | Backup-Module | Datei-Kopie |
| Betriebsaufwand | Gering (1 Binary) | Gering (1 Service) | Minimal | Gering (1 Binary) | Minimal |

**Entscheidung: Qdrant** -- Nativer Hybrid Search (Dense + Sparse Vektoren) mit BM25, tiefe Integration von Metadata-Filtering in den Suchpfad, einfacher Betrieb als Single Rust Binary. Die Payload-System unterstuetzt beliebige JSON-Metadaten inkl. Geo-Koordinaten.

**Starke Alternative: pgvector + pgvectorscale + pg_textsearch** -- Falls komplexe relationale Metadaten-Queries (JOINs ueber Autoren/Tags/Orte) wichtiger sind als die Retrieval-Performance. PostgreSQL bietet die maechtigsten Metadaten-Queries.

**Reddit-Community-Konsens:** "Qdrant ist die beste Open-Source Performance-Option" / "We LOVE Qdrant!" -- Bei >50K Vektoren mit Filtering klar vor ChromaDB. pgvector fuer 90% der Faelle "gut genug", aber Qdrant besser fuer dedizierte Vektor-Suche.

### Konfiguration

- **Deployment:** Docker ARM64 auf Mac Mini
- **Speicher:** ~2-5 GB RAM fuer 100K Dokumente mit Metadaten
- **Hybrid Search:** Dense Vektoren (BGE-M3) + Sparse Vektoren (BM25) in einer Collection
- **Fusion:** Reciprocal Rank Fusion (RRF) oder Depth-Biased Score Fusion (DBSF)
- **Payload-Indizes:** Keyword (tags, platform), Datetime (date), Geo (locations), Fulltext (content)

---

## 2. RAG Framework: LlamaIndex

### Warum LlamaIndex?

| Kriterium | LlamaIndex | Haystack | LangChain | RAGFlow |
|---|---|---|---|---|
| Chunking-Strategien | A+ (hierarchisch, auto-merge) | B+ | B | A |
| Retrieval-Qualitaet | A (hybrid, reranking, multi-query) | A | B+ | A |
| Source Attribution | A (native Citations) | A- | B+ | A+ |
| Metadata/Entities | A (Property Graph Index) | B+ | B+ | A- |
| Erweiterbarkeit | A+ (reines Framework) | A | A | B (Applikation) |
| 300+ Konnektoren | Ja (LlamaHub) | Nein | Ja (variabel) | Nein |

**Entscheidung: LlamaIndex** -- Bestes Framework fuer minimalen Informationsverlust dank hierarchischem Chunking mit Auto-Merging Retrieval. Kleine Leaf-Chunks werden fuer praezise Embedding erstellt, aber bei Retrieval automatisch zu Parent-Chunks expandiert wenn mehrere Children matchen. 35% Retrieval-Accuracy-Boost dokumentiert.

**Reddit-Community-Konsens:** "LlamaIndex ist objektiv ueberlegen fuer komplexes Document-Parsing" / "LangChain ist fuer simples RAG ueberengineered -- 2.7x teurer durch versteckte interne API-Calls". Trend: LlamaIndex fuer Indexing+Retrieval, LangGraph nur falls Agent-Workflows benoetigt.

### Schluessel-Features fuer dieses Projekt

1. **Hierarchical Node Parser:** Erstellt Chunk-Baum (512 -> 1024 -> 2048 Tokens)
2. **Auto-Merging Retriever:** Ersetzt Leaf-Chunks automatisch durch Parents bei Mehrfach-Hits
3. **Property Graph Index:** Verbindet Entities (Personen, Orte, Themen) als Graph
4. **Citation Query Engine:** Generiert Antworten mit praezisen Quellen-Verweisen
5. **LlamaHub Konnektoren:** YouTube, Reddit, Web, PDF direkt unterstuetzt

---

## 3. Embedding Model: BGE-M3

### Warum BGE-M3?

| Modell | MTEB Multi. | Params | Dims | Context | Matryoshka | Lizenz | Triple-Retrieval |
|---|---|---|---|---|---|---|---|
| **BGE-M3** | 63.0 | 568M | 1024 | 8192 | Nein | MIT | Ja (Dense+Sparse+ColBERT) |
| Qwen3-Embed-8B | 70.58 | 8B | Flex | 8192 | Ja | Apache 2.0 | Nein |
| Qwen3-Embed-0.6B | ~62-64 | 0.6B | Flex | 8192 | Ja | Apache 2.0 | Nein |
| Jina v3 | ~60-62 | 570M | 1024 | 8192 | Ja | CC-BY-NC | Nein |
| Nomic v2 MoE | ~58-60 | 475M | 768 | **512** | Ja | Apache 2.0 | Nein |

**Entscheidung: BGE-M3** trotz niedrigerem MTEB-Score als Qwen3-8B, weil:

1. **Triple-Retrieval** (Dense + Sparse + Multi-Vector/ColBERT) ist einzigartig und perfekt fuer Qdrant's Hybrid Search. Deutsche Komposita wie "Handschuhfach" profitieren enorm vom Sparse/BM25-Modus.
2. **Geschwindigkeit:** Bei 568M Params werden 100K Dokumente in Minuten bis wenigen Stunden embedded -- nicht Tagen wie beim 8B-Modell. Kritisch fuer Iterations-Geschwindigkeit beim Tuning.
3. **Battle-tested:** Das am weitesten verbreitete multilinguale Embedding-Modell mit extensiver Community.
4. **MIT Lizenz:** Keinerlei Einschraenkungen.
5. **MIRACL 70.0:** Auf dem multilingualen Retrieval-Benchmark der fuer RAG am relevantesten ist, unter den Besten.

**Best Practice: NICHT uebersetzen vor dem Embedding.** Moderne multilinguale Modelle lernen einen gemeinsamen semantischen Raum. "Hund" (DE) und "dog" (EN) liegen nah beieinander. Eine deutsche Query findet englische Dokumente und umgekehrt.

### Upgrade-Pfad

Spaeter optional: Qwen3-Embedding-8B fuer Dokument-Embedding (Qualitaet), BGE-M3 fuer Query-Embedding (Geschwindigkeit). Asymmetrisches Embedding ist mit Instruction-Aware Modellen moeglich.

---

## 4. Lokales LLM: Qwen 2.5-72B-Instruct

### Warum Qwen statt MiniMax?

| Kriterium | Qwen 2.5-72B | MiniMax M2 | Command-R+ 104B | Llama 3.3-70B |
|---|---|---|---|---|
| Deutsch | Exzellent (29+ Sprachen) | Schwach (Code-fokussiert) | Mittel | Gut |
| RAG-Grounding | Gutes Structured Output | Kein spezieller Support | Eingebaute Citations | Gutes Structured Output |
| Source Citations | Via Prompting | Via Prompting | Nativ (Grounding Spans) | Via Prompting |
| Speed (Q8, Mac Ultra) | ~25-35 tok/s | ~30 tok/s (Q4) | ~18 tok/s (Q5) | ~25-35 tok/s |
| RAM-Verbrauch (Q8) | ~75 GB | ~70 GB (Q4) | ~85 GB (Q6) | ~73 GB |
| Verbleibendes RAM | ~437 GB | ~442 GB | ~427 GB | ~439 GB |

**Entscheidung: Qwen 2.5-72B-Instruct @ Q8_0** -- Beste deutsche Sprachqualitaet aller Open-Source Modelle, exzellentes Structured Output fuer Source Citations, 128K Context Window. Bei Q8 Quantisierung (99% Qualitaetserhalt) nur ~75 GB RAM, laesst ~437 GB fuer KV-Cache, Reranker und Embeddings.

**MiniMax-Bewertung:** MiniMax M2/M2.5 ist beeindruckend, aber auf Coding und Agentic Workflows optimiert -- nicht auf multilinguales RAG. Die "Multi-Language"-Staerke bezieht sich primaer auf Programmiersprachen (Rust, Java, Go), nicht Deutsch/Englisch.

**Upgrade-Option:** Qwen3-235B-A22B (MoE, 235B total / 22B aktiv) bei Q5_K_M (~130 GB). Passt komfortabel, ~30 tok/s, maximale Intelligenz. Spaeter testen.

### Inference Framework

- **Primaer: MLX** -- 20-50% schneller als Ollama auf Apple Silicon, bester Prompt-Processing Speed (kritisch fuer RAG wo viele Dokumente eingefuettert werden)
- **Sekundaer: Ollama** -- Fuer API-Kompatibilitaet mit LlamaIndex und einfaches Modell-Management. OpenAI-kompatible API out of the box.

### Reranker: Qwen3-Reranker-8B

- Top-ranked auf MTEB multilingual
- 32K Context Window
- ~10 GB RAM -- trivial neben dem Haupt-LLM
- **15-40% Verbesserung** der Retrieval-Accuracy ueber Semantic Search allein
- Optimal: Top 50-75 Kandidaten reranken, dann Top 5 an LLM

---

## 5. Ingestion Pipeline

### Architektur mit Prefect

```
Taeglicher Cron (via Prefect auf Mac Mini)
  |
  +-- YouTube Flow
  |     youtube-transcript-api -> Whisper Fallback (mlx-whisper) -> Chunk bei Kapiteln
  |
  +-- Twitter/X Flow
  |     Twikit (mit Account-Pool) -> Threads extrahieren -> Tweet-Level Chunks
  |
  +-- Reddit Flow
  |     PRAW (OAuth) -> Posts + Comment Trees -> Hierarchische Chunks
  |
  +-- Web/RSS Flow
  |     Trafilatura (Artikel) -> Crawl4AI (JS-Seiten) -> Semantische Chunks
  |
  +-- PDF Flow
  |     Docling (komplex: Tabellen, Layouts) / PyMuPDF4LLM (einfach) -> Hierarchische Chunks
  |
  +-- [Alle Flows vereinen] -> Metadata-Normalisierung
  |     Schema: {source, author, date, url, platform, language, content, chunks}
  |
  +-- NER + Tagging (auf Mac Studio Ultra)
  |     spaCy -> GLiNER (Entities) -> BERTopic (Topics) -> Neo4j (Graph)
  |
  +-- Embedding (auf Mac Studio Ultra)
  |     BGE-M3 via sentence-transformers (MPS-beschleunigt)
  |
  +-- Upsert
        Qdrant (Vektoren + Sparse + Metadaten)
        Neo4j (Entities + Beziehungen)
        PostgreSQL (Dokument-Metadaten + Source-Links)
```

### Tool-Auswahl pro Medientyp

| Medientyp | Primaer-Tool | Fallback | Metadaten |
|---|---|---|---|
| **PDF (komplex)** | Docling (IBM) -- 97.9% Accuracy auf Tabellen | MinerU | Filename, Pages, Sections |
| **PDF (einfach)** | PyMuPDF4LLM -- ~520 Seiten/Sek | Marker | Filename, Pages |
| **YouTube** | youtube-transcript-api | yt-dlp + mlx-whisper | Channel, Video-ID, Timestamps, Kapitel |
| **Twitter/X** | Twikit (GraphQL API) | twscrape | Author, Tweet-ID, Thread-ID, Likes, Date |
| **Reddit** | PRAW (100 req/min, OAuth) | asyncpraw | Subreddit, Author, Score, Comment-Tree |
| **Web-Artikel** | Trafilatura (F1: 0.958) | Crawl4AI (JS-Seiten) | Domain, URL, Author, Date |
| **JS-heavy Seiten** | Crawl4AI (Playwright) | Firecrawl (self-hosted) | Domain, URL |

### Chunking-Strategie (Minimaler Informationsverlust)

| Medientyp | Strategie | Chunk-Groesse |
|---|---|---|
| **YouTube** | Kapitel-Grenzen, sonst 60-Sek-Fenster mit Overlap | ~500-1000 Tokens |
| **Twitter** | Jeder Tweet = Chunk, Thread = Parent-Dokument | Natuerlich |
| **Reddit** | Post = ein Chunk, Top-Comments = einzelne Chunks, verlinkt zum Parent | Natuerlich |
| **Web-Artikel** | Semantisches Chunking (Embedding-Similarity Breakpoints) | ~500 Tokens |
| **PDFs** | **Hierarchisch (Parent-Child)** via LlamaIndex: Leaf (512) -> Parent (1024) -> Grandparent (2048) | 512/1024/2048 |

**Kritischer Community-Hinweis:** "Chunking-Qualitaet bestimmt die Retrieval-Accuracy mehr als die Modell-Wahl." Optimiertes semantisches Chunking erreicht Faithfulness-Scores von 0.79-0.82 vs. 0.47-0.51 bei naivem Chunking.

---

## 6. NER, Tagging & Knowledge Graph

### NER-Pipeline

```
Dokument -> spaCy (Tokenisierung/Satz-Segmentierung)
         -> GLiNER (Zero-Shot NER: PER, ORG, LOC, TOPIC, Custom-Typen)
         -> Sprach-Erkennung (fasttext/langdetect)
         -> Optional: Lokales LLM fuer komplexe Entity-Disambiguierung
```

**GLiNER** ist die Top-Wahl weil:
- Zero-Shot: Entity-Typen zur Laufzeit definieren ohne Retraining
- Multilingual (DE/EN/FR/ES/IT/PT trainiert)
- Performance vergleichbar mit ChatGPT bei Bruchteil der Compute-Kosten
- Direkt in spaCy integrierbar via `gliner-spacy`
- ONNX-Support fuer Beschleunigung

### Topic Modeling: BERTopic

- **Hierarchisches Topic Modeling** via `.hierarchical_topics()` -- baut automatisch Topic-Taxonomie
- **Multilingual** via `paraphrase-multilingual-MiniLM-L12-v2` Embeddings
- **Inkrementelle Updates** via `.merge_models()` -- neue Dokument-Batches einfach integrieren
- **34%+ besser** als LDA und Top2Vec in Clustering-Qualitaet
- **LLM-basierte Labels** via Ollama fuer menschenlesbare Topic-Namen

### Knowledge Graph: Neo4j + LightRAG

**Neo4j** fuer strukturierte Entity-Beziehungen:

```
(:Document {id, title, url, platform, date, language})
(:Person {name, aliases[]})
(:Organization {name, type})
(:Location {name, geo_coordinates, country})
(:Topic {name, hierarchy_path, bertopic_id})
(:Platform {name})  // YouTube, Twitter, Reddit, Web, PDF

(:Document)-[:MENTIONS {confidence, span}]->(:Person)
(:Document)-[:MENTIONS {confidence, span}]->(:Organization)
(:Document)-[:MENTIONS {confidence, span}]->(:Location)
(:Document)-[:HAS_TOPIC {probability}]->(:Topic)
(:Document)-[:FROM_PLATFORM]->(:Platform)
(:Person)-[:AFFILIATED_WITH]->(:Organization)
(:Topic)-[:SUBTOPIC_OF]->(:Topic)
```

**LightRAG** (EMNLP 2025) fuer Graph-augmentiertes Retrieval:
- Dual-Level Retrieval: Low-Level (Entities) + High-Level (Themen)
- Query-Kosten: <100 Tokens vs. 610K bei Microsoft GraphRAG
- Inkrementelle Graph-Updates via Graph Union (~50% schneller als Full Reindex)
- **Wichtig:** Mindestens 32K Context-Window in Ollama konfigurieren (`PARAMETER num_ctx 32768`)

**Community-Warnung zu GraphRAG:** "Nicht mit GraphRAG starten. Nur hinzufuegen wenn simples RAG bei Multi-Hop-Fragen versagt." Indexing-Kosten bei 100K Dokumenten mit lokalem LLM: Stunden bis Tage.

### Provenance-Schema (Bidirektionale Source-Links)

```python
{
    "entity": "Angela Merkel",
    "entity_type": "PERSON",
    "source_document_id": "doc_abc123",
    "source_platform": "youtube",
    "source_url": "https://youtube.com/watch?v=...",
    "chunk_id": "chunk_456",
    "char_span": [142, 156],
    "extraction_confidence": 0.94,
    "extraction_method": "gliner_multilingual_v2",
    "extraction_timestamp": "2026-02-16T10:30:00Z"
}
```

**Beispiel-Query in Neo4j:**
```cypher
// "Zeige alle Dokumente von Autor X ueber Thema Y von Plattform Z"
MATCH (d:Document)-[:MENTIONS]->(p:Person {name: "Autor X"}),
      (d)-[:HAS_TOPIC]->(t:Topic {name: "Thema Y"}),
      (d)-[:FROM_PLATFORM]->(pl:Platform {name: "YouTube"})
RETURN d.title, d.url, d.date
ORDER BY d.date DESC
```

---

## 7. Hardware-Aufteilung

### LXC Container auf Proxmox (192.168.178.4, x86, keine GPU)

Alles ausser LLM-Inference laeuft hier:

| Service | RAM-Schaetzung | Aufgabe |
|---|---|---|
| Qdrant (Docker x86) | 2-5 GB | Vektor-Datenbank |
| Neo4j (Docker x86) | 2-4 GB | Knowledge Graph |
| PostgreSQL (Docker x86) | 1-2 GB | Metadaten-DB |
| Prefect Server | 0.5-1 GB | Pipeline-Orchestrierung |
| BGE-M3 Embedding (CPU) | ~3-4 GB | Embedding (sentence-transformers, CPU-Modus) |
| GLiNER + spaCy (CPU) | ~2-3 GB | NER |
| BERTopic (CPU) | ~3-5 GB | Topic Modeling |
| Python App (FastAPI, CLI) | ~1-2 GB | Anwendung |
| **Gesamt** | **~15-26 GB** | |

**Empfohlene LXC-Konfiguration:** 32 GB RAM, 8 vCPUs, 100 GB Storage (SSD).

### Mac Studio Ultra 512 GB (Dual-Model LLM + Reranker)

Zwei LLMs gleichzeitig + Reranker -- erreichbar via Ollama REST API im LAN:

| Service | RAM-Schaetzung | Aufgabe |
|---|---|---|
| **Qwen 2.5-72B Q8** | ~75 GB | Kern-RAG: Generation, Citations, Deutsche Texte, JSON-Output |
| **MiniMax M2.5 Q4_K_M** | ~138 GB | Agentic Router: Query-Dekomposition, Tool-Orchestrierung, Code-RAG, >128K Kontext |
| Qwen3-Reranker-8B | ~10 GB | Reranking |
| KV-Cache (beide Modelle) | ~50-80 GB | Fuer lange RAG-Kontexte |
| **Gesamt** | **~273-303 GB** | Von 512 GB |

**Verbleibend:** ~209-239 GB fuer KV-Cache-Erweiterung und Experimente.

### Dual-Model Routing-Strategie

MiniMax M2.5 hat eine **88% Halluzinationsrate** (AA-Omniscience) -- zu hoch fuer direkte RAG-Antworten. Aber mit **76.8% BFCL Function Calling** (industrie-best) und **200K Context Window** ist es ideal als Orchestrator.

```
User Query
    |
    v
[Query-Klassifikation im LXC -- regelbasiert oder leichtgewichtig]
    |
    +-- Einfache Frage / Standard-RAG?
    |     --> Qdrant Hybrid Search
    |     --> Qwen3-Reranker (Top 50 -> Top 5)
    |     --> Qwen 2.5-72B: Antwort mit Citations
    |
    +-- Multi-Step / Komplexe Recherche?
    |     --> MiniMax M2.5 als Agentic Router:
    |           1. Zerlegt Query in Sub-Queries
    |           2. Entscheidet: Qdrant, Neo4j, oder beides?
    |           3. Ruft Tools/DBs auf (Function Calling)
    |           4. Sammelt Ergebnisse
    |     --> Qwen 2.5-72B: Finale Synthese + Citations (!)
    |
    +-- Code-bezogene Fragen?
    |     --> MiniMax M2.5 direkt (SWE-Bench 80.2%)
    |
    +-- Sehr langer Kontext (>128K Tokens)?
          --> MiniMax M2.5 (200K Window)
```

**Wichtig:** Auch wenn M2.5 orchestriert, generiert **Qwen immer die finale Antwort** -- wegen niedrigerer Halluzinationsrate und besserem Deutsch.

### Warum Dual-Model statt nur Qwen?

| Kriterium | Nur Qwen 2.5-72B | Qwen + MiniMax M2.5 |
|---|---|---|
| Einfache RAG-Queries | Gleich gut | Gleich gut (Qwen antwortet) |
| Multi-Step Recherche | Gut, aber linear | M2.5 orchestriert parallel, Qwen synthetisiert |
| Function Calling | Gut | 76.8% BFCL (M2.5) -- deutlich besser |
| Context Window | 128K | 200K (M2.5 fuer lange Kontexte) |
| Code-Fragen | Gut | 80.2% SWE-Bench (M2.5) |
| RAM-Overhead | ~85 GB | ~223 GB (+138 GB) -- passt in 512 GB |

### Netzwerk-Kommunikation

```
LXC Container (192.168.178.x)
  |
  |-- Ollama API --> Mac Studio Ultra (192.168.178.y:11434)
  |     - POST /api/generate (model=qwen2.5:72b)     -- Kern-RAG
  |     - POST /api/generate (model=minimax-m2.5)     -- Agentic Router
  |     - POST /api/embed    (model=bge-m3)           -- Bulk-Embedding Fallback
  |
  |-- Lokal im LXC:
        - Qdrant:    localhost:6333
        - Neo4j:     localhost:7687
        - PostgreSQL: localhost:5432
        - Prefect:   localhost:4200
```

---

## 8. Geschaetzte Performance

| Metrik | LXC (CPU, x86) | Mac Studio Ultra | Bemerkung |
|---|---|---|---|
| Embedding (100K Docs, BGE-M3) | 6-12 Stunden (CPU) | 1-3 Stunden (MPS) | Bulk-Import ggf. via Mac Studio |
| Taegliche Ingestion (100-500 Docs) | 10-45 Minuten | -- | CPU ausreichend fuer Daily |
| NER mit GLiNER | 50-200 Docs/Min (CPU) | 100-500 Docs/Min (MPS) | CPU akzeptabel |
| BERTopic Training (100K Docs) | 1-3 Stunden (CPU) | 30-60 Min (MPS) | Einmalig + inkrementell |
| LLM Generation (Qwen) | -- | ~30-40 tok/s (Q8) | Kern-RAG Antworten |
| LLM Generation (MiniMax) | -- | ~20-25 tok/s (Q4) | Agentic Routing, Code |
| Reranking (Top 50) | -- | <500ms | Nur auf Mac Studio |
| Retrieval-Latenz (Qdrant) | <100ms | -- | Im LXC |
| End-to-End Query (einfach) | ~5-10 Sekunden | -- | Qdrant -> Reranker -> Qwen |
| End-to-End Query (komplex) | ~10-20 Sekunden | -- | MiniMax orchestriert -> Qwen synthetisiert |

---

## 9. Risiken und Mitigationen

| Risiko | Mitigation |
|---|---|
| **Twitter/X Scraping bricht** | Fragil. Twikit Account-Pool + Retry-Logik. Monitoring. Zweit-Tool twscrape. |
| **Chunking-Verluste** | Hierarchisches Chunking + Auto-Merging Retrieval. Evaluation mit echten Queries. |
| **GraphRAG Indexing-Kosten** | Mit LightRAG starten (nicht Microsoft GraphRAG). Nur bei Bedarf. |
| **Modell-Updates** | BGE-M3 und Qwen sind aktiv maintained. Embedding-Wechsel erfordert Reindexierung. |
| **Skalierung ueber 100K** | Qdrant skaliert bis Milliarden. Engpass waere eher Ingestion-Pipeline. |

---

## 10. Implementierungs-Reihenfolge (Vorschlag)

### Phase 0: Infrastruktur (1-2 Tage)
0. LXC Container auf Proxmox (192.168.178.4) erstellen: Ubuntu 24.04, 32GB RAM, 8 vCPU, 100GB SSD
1. Docker + Docker Compose im LXC installieren
2. Ollama auf Mac Studio Ultra: Qwen 2.5-72B + MiniMax M2.5 + Reranker laden + konfigurieren
3. Netzwerk-Konnektivitaet LXC <-> Mac Studio testen (beide Modelle erreichbar)

### Phase 1: Kern-RAG (2-3 Wochen)
4. Docker Compose im LXC: Qdrant + Neo4j + PostgreSQL
5. Python-Projekt mit uv im LXC initialisieren
6. BGE-M3 Embedding im LXC (CPU-Modus) einrichten
7. LlamaIndex mit hierarchischem Chunking aufsetzen
8. Ollama-Anbindung ans Mac Studio testen (LAN API)
9. Einfache PDF + Web Ingestion (Docling + Trafilatura)
10. Chat/Q&A mit Source Citations testen

### Phase 2: Multi-Source Ingestion (2-3 Wochen)
11. YouTube Transcript Pipeline
12. Reddit Pipeline (PRAW)
13. Twitter/X Pipeline (Twikit)
14. Prefect im LXC fuer taegliche Orchestrierung
15. Unified Metadata Schema

### Phase 3: Tagging & Knowledge Graph (2-3 Wochen)
16. GLiNER NER-Pipeline (CPU im LXC)
17. BERTopic Topic Modeling (CPU im LXC)
18. Neo4j Graph-Schema + Population
19. Bidirektionale Source-Links
20. LightRAG fuer Graph-augmentiertes Retrieval

### Phase 4: UI & Polish (1-2 Wochen)
21. Such-Interface (Streamlit oder Gradio)
22. Knowledge-Graph-Exploration (Neo4j Browser oder custom)
23. Evaluation mit echten Queries
24. Performance-Tuning

---

## 11. Vergleich mit Cognee

### Was ist Cognee?

Cognee (topoteretes/cognee, Apache 2.0, ~6.800 GitHub Stars, v0.5.2 pre-1.0) ist eine "AI Memory Engine" die automatisch Knowledge Graphs aus unstrukturierten Daten via LLM-basierter Entity-Extraktion baut. Kernkonzept: ECL-Pipeline (Extract, Cognify, Load) -- in 3 Zeilen Code ein funktionierendes GraphRAG.

### Feature-Vergleich

| Kriterium | Cognee | Unser Custom Stack |
|---|---|---|
| **Setup-Aufwand** | Minuten (pip install + 6 Zeilen) | Wochen (24 Tasks, 8 Phasen) |
| **Knowledge Graph** | Automatisch via LLM | Manuell via GLiNER + Neo4j (volle Kontrolle) |
| **Entity Extraction** | LLM-basiert (teuer, langsam, reichere Beziehungen) | GLiNER (schnell, kostenlos, offline, zuverlaessig) |
| **Topic Modeling** | Nicht vorhanden | BERTopic (hierarchisch, 34%+ besser als LDA) |
| **Hybrid Search (BM25+Dense)** | Nein -- nur Vector + Graph | Ja -- BGE-M3 Dense+Sparse+ColBERT via Qdrant |
| **Embedding Multi-Vector** | Nein (nur Dense) | Ja (BGE-M3 Triple-Retrieval) |
| **YouTube/Twitter/Reddit** | Nicht eingebaut | Dedizierte Pipelines |
| **PDF Parsing** | Basis (30+ Dateitypen) | Docling (97.9% Tabellen-Accuracy) + PyMuPDF4LLM |
| **Chunking-Kontrolle** | Fix 4K Tokens (Ollama), kaum konfigurierbar | Hierarchisch (512/1024/2048) + Semantisch + Medien-spezifisch |
| **Skalierung 100K Docs** | Problematisch (LLM pro Chunk = Tage lokal) | Gut (GLiNER: 100-500 Docs/Min) |
| **Lokale LLMs** | Braucht 32B+ fuer zuverlaessige JSON-Ausgabe | GLiNER braucht kein LLM fuer NER |
| **Orchestrierung** | Eigene Pipeline (einfach, begrenzt) | Prefect (Scheduling, Retry, Monitoring) |
| **Reranking** | Nicht eingebaut | Qwen3-Reranker-8B (+15-40% Accuracy) |
| **API-Stabilitaet** | Pre-1.0 (v0.5.2), Breaking Changes moeglich | Jede Komponente einzeln battle-tested |
| **Community** | ~6.8K Stars, kaum Reddit/HN-Praesenz | Jede Komponente hat grosse Community |
| **Qdrant-Integration** | Community-maintained (bekannte Bugs) | Direkte Qdrant API |

### Cognee-Staerken

1. **Schneller Prototyp** -- GraphRAG in Minuten statt Wochen
2. **Automatische Beziehungs-Extraktion** -- LLM erkennt komplexe Beziehungen ohne manuelle Regeln
3. **Multi-Hop Reasoning** -- Graph-basierte Suche ueber mehrere Dokumente
4. **Eingebautes Provenance-Tracking**

### Cognee-Schwaechen fuer unseren Use Case

1. **Skalierung:** LLM-pro-Chunk Extraktion bei 100K Docs = Tage/Wochen lokal oder tausende Dollar Cloud-API
2. **Kein Topic Modeling** -- Feature das wir brauchen existiert nicht
3. **Kein Hybrid Search (BM25)** -- Cognee's "Hybrid" ist Vector+Graph, nicht Vector+BM25
4. **Kein Social Media Ingestion** -- YouTube, Twitter, Reddit nicht eingebaut
5. **4K Token Chunk-Limit mit Ollama** -- Nicht konfigurierbar
6. **Modelle unter 32B produzieren "noisy graphs"** (unabhaengig dokumentiert)
7. **Tutorial kaputt bei Neuinstallation** (GitHub Issue #1557)
8. **Qdrant-Adapter: Community-maintained mit Bugs** (Issue #1756)
9. **Reddit/HN-Praesenz: Null** -- Keine einzige organische Community-Diskussion gefunden
10. **Benchmarks nur selbst-publiziert** (24 Fragen, keine unabhaengige Validierung)

### Entscheidung

**Fuer unseren Use Case (100K Docs, 5 Medientypen, lokal/LXC, deutsch+englisch, Topic Modeling, Hybrid Search): Der Custom Stack ist klar ueberlegen.**

**Moeglicher spaeterer Hybrid-Ansatz:** Cognee als optionale Ergaenzung fuer Multi-Hop Reasoning evaluieren (via LlamaIndex-Integration), waehrend der Custom Stack die Schwerlast uebernimmt.

### Quellen (Cognee)

- [Cognee GitHub](https://github.com/topoteretes/cognee)
- [Cognee Dokumentation](https://docs.cognee.ai/)
- [Cognee AI Memory Benchmarks](https://www.cognee.ai/blog/deep-dives/ai-memory-evals-0825)
- [Self-Hosting Cognee mit Ollama (unabhaengiger Review)](https://www.glukhov.org/post/2025/12/selfhosting-cognee-quickstart-llms-comparison/)
- [Cognee + LlamaIndex GraphRAG](https://www.analyticsvidhya.com/blog/2025/02/cognee-llamaindex/)
- [GitHub Issue #1557 -- Tutorial kaputt](https://github.com/topoteretes/cognee/issues/1557)
- [GitHub Issue #1756 -- Qdrant-Adapter Bugs](https://github.com/topoteretes/cognee/issues/1756)

---

## Quellen (Auswahl der wichtigsten)

### Vektor-Datenbanken
- [Qdrant Hybrid Search](https://qdrant.tech/articles/hybrid-search/)
- [Qdrant ARM Architecture](https://qdrant.tech/blog/qdrant-supports-arm-architecture/)
- [pgvectorscale GitHub](https://github.com/timescale/pgvectorscale)

### RAG Frameworks
- [LlamaIndex Hierarchical Node Parser](https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/hierarchical/)
- [15 Best Open-Source RAG Frameworks 2026](https://www.firecrawl.dev/blog/best-open-source-rag-frameworks)
- [LangChain vs LlamaIndex 2025](https://latenode.com/blog/langchain-vs-llamaindex-2025-complete-rag-framework-comparison)

### Embedding Models
- [BGE-M3 HuggingFace](https://huggingface.co/BAAI/bge-m3)
- [Qwen3-Embedding Blog](https://qwenlm.github.io/blog/qwen3-embedding/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

### Ingestion
- [Docling GitHub](https://github.com/docling-project/docling)
- [Trafilatura Evaluation](https://trafilatura.readthedocs.io/en/latest/evaluation.html)
- [Crawl4AI vs Firecrawl](https://blog.apify.com/crawl4ai-vs-firecrawl/)
- [Twikit GitHub](https://github.com/d60/twikit)
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)

### NER & Knowledge Graphs
- [GLiNER GitHub](https://github.com/urchade/GLiNER)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [Neo4j Knowledge Graph RAG](https://neo4j.com/blog/developer/knowledge-graph-rag-application/)

### Lokale LLMs
- [Qwen 2.5 Technical Report](https://qwenlm.github.io/blog/qwen2.5/)
- [Apple Silicon LLM Inference (arxiv)](https://arxiv.org/abs/2511.05502)
- [Best Local LLMs for Mac 2026](https://www.insiderllm.com/guides/best-local-llms-mac-2026/)

### Community/Reddit
- [RAG Best Practices from 100+ Teams](https://www.kapa.ai/blog/rag-best-practices)
- [Six Lessons Building RAG in Production](https://towardsdatascience.com/six-lessons-learned-building-rag-systems-in-production/)
- [Is LangChain Becoming Too Complex?](https://github.com/orgs/community/discussions/182015)
