# RAG Wissensmanagement-System -- Setup & Betrieb

## Architektur-Uebersicht

```
LXC Container "rag" (192.168.178.182)
  Ubuntu 24.04 / 8 Cores / 32GB RAM / 100GB Disk

  Docker Compose:
    Qdrant  :6333    (Vektordatenbank, Hybrid Search)
    Neo4j   :7474    (Knowledge Graph)
    Postgres :5432   (Dokument-Metadaten)

  RAG Python App:
    BGE-M3 Embedding (CPU, ~2.3GB Modell)
    GLiNER NER       (CPU, ~500MB Modell)
    BERTopic         (CPU)
    CLI + FastAPI
        |
        | LAN (REST API)
        |
Mac Studio Ultra (512GB RAM)
  OpenAI-kompatible API :54321 (LM Studio / mlx-server)
    MiniMax M2.5 8bit    (~138GB)  RAG + Agent
    Spaeter: Qwen 2.5-72B, Qwen3-Reranker
```

---

## Was ist bereits eingerichtet

### LXC Container (Proxmox ID 110)

- **Hostname:** rag
- **IP:** 192.168.178.182 (DHCP)
- **SSH:** `ssh root@192.168.178.182` (Key-basiert, Passwort: ragserver2026)
- **OS:** Ubuntu 24.04 LTS
- **Ressourcen:** 8 Cores, 32GB RAM, 4GB Swap, 100GB Disk
- **Features:** nesting=1, keyctl=1, apparmor=unconfined (fuer Docker-in-LXC)
- **Autostart:** ja (startet bei Proxmox-Boot)

### Installierte Software

| Software | Version | Zweck |
|----------|---------|-------|
| Python | 3.12.3 | Runtime |
| uv | 0.10.3 | Paketmanager |
| Docker | 29.2.1 | Container-Runtime |
| Docker Compose | 5.0.2 | Service-Orchestrierung |
| Node.js | 22.22.0 | Claude Code |
| Claude Code | 2.1.42 | AI-Entwicklungsassistent |
| tmux | vorhanden | Session-Management |
| Git | vorhanden | Versionierung |

### Docker Services (laufen automatisch)

| Service | Image | Port | Daten |
|---------|-------|------|-------|
| Qdrant | qdrant/qdrant:latest | 6333, 6334 | Docker Volume `rag_qdrant_data` |
| Neo4j | neo4j:5-community | 7474 (Web), 7687 (Bolt) | Docker Volume `rag_neo4j_data` |
| PostgreSQL | postgres:16-alpine | 5432 | Docker Volume `rag_postgres_data` |

**Zugangsdaten (Standard):**
- Neo4j: `neo4j` / `changeme`
- PostgreSQL: `rag` / `changeme` / DB: `rag`

### RAG Application

**Projektverzeichnis:** `/root/rag`

**Python-Module:**

```
src/rag/
  config.py          # Pydantic Settings (.env)
  models.py          # Document, Chunk, Entity, Topic, Platform
  storage/
    qdrant.py        # Hybrid Search (Dense + Sparse)
    postgres.py      # Dokument-Metadaten CRUD
    neo4j_store.py   # Knowledge Graph (Entities, Relationships)
  ingestion/
    base.py          # BaseIngestor ABC
    pdf.py           # Docling + PyMuPDF4LLM Fallback
    youtube.py       # youtube-transcript-api (60s Fenster)
    web.py           # Trafilatura Artikel-Extraktion
    reddit.py        # PRAW (OAuth erforderlich)
    twitter.py       # Twikit (Session-Cookies erforderlich)
  processing/
    embedding.py     # BGE-M3 (Dense 1024d + Sparse)
    chunking.py      # Hierarchisch (512/1024/2048) + Medien-spezifisch
    ner.py           # GLiNER Zero-Shot (PERSON, ORG, LOCATION, TOPIC, EVENT)
    topics.py        # BERTopic (multilingual, inkrementell)
    graph_builder.py # NER + Topics -> Neo4j
  retrieval/
    hybrid.py        # BGE-M3 Embed -> Qdrant Search
    reranker.py      # Qwen3-Reranker (deaktiviert bis Modell geladen)
  generation/
    llm.py           # OpenAI-kompatibler LLM Client (mit Retry-Logik)
    router.py        # Query Router
    citation.py      # Citation-aware Antworten mit [1], [2] Referenzen
  pipeline/
    sources.py       # YAML-Parser fuer sources.yaml
    dedup.py         # URL-Dedup gegen PostgreSQL
    tasks.py         # Prefect Tasks (Web, YouTube, Reddit, Twitter)
    flows.py         # Prefect Flow (daily-ingestion)
    deploy.py        # Prefect Deployment + Worker
  cli.py             # Typer CLI
  api.py             # FastAPI REST API
```

**Heruntergeladene ML-Modelle (auf dem LXC, unter ~/.cache):**

| Modell | Groesse | Zweck |
|--------|---------|-------|
| BAAI/bge-m3 | ~2.3 GB | Embedding (Dense + Sparse) |
| urchade/gliner_multi-v2.1 | ~500 MB | NER (Zero-Shot) |
| paraphrase-multilingual-MiniLM-L12-v2 | ~500 MB | BERTopic Embedding |

**Tests:** 55 Tests bestehen (ohne Network-Tests)

---

## Fehlende Schritte

### Schritt 1: LLM auf Mac Studio einrichten [ERLEDIGT]

MiniMax M2.5 (8bit) laeuft via LM Studio auf dem Mac Studio (.8), Port 54321.
OpenAI-kompatible API unter: `http://192.168.178.8:54321/v1`

Spaeter geplant: Qwen 2.5-72B (RAG) + Qwen3-Reranker via LM Studio.

### Schritt 2: .env Datei konfigurieren [ERLEDIGT]

.env erstellt aus .env.example. Konfigurierte Werte:

```env
LLM_BASE_URL=http://192.168.178.8:54321/v1
LLM_MODEL_RAG=mlx-community/MiniMax-M2.5-8bit
LLM_MODEL_AGENT=mlx-community/MiniMax-M2.5-8bit
LLM_MODEL_RERANKER=
```

### Schritt 3: Netzwerk-Konnektivitaet testen [ERLEDIGT]

```bash
# OpenAI-kompatible API erreichbar?
curl -s http://192.168.178.8:54321/v1/models | python3 -m json.tool

# Chat-Completion Test
curl -s http://192.168.178.8:54321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/MiniMax-M2.5-8bit","messages":[{"role":"user","content":"Sag Hallo"}],"max_tokens":20}'

# Test via RAG-Code
cd /root/rag && source .venv/bin/activate
python -c "
from rag.generation.llm import LLMClient
client = LLMClient()
print('LLM erreichbar:', client.is_available())
print('MiniMax verfuegbar:', client.is_available('MiniMax'))
"
```

### Schritt 4: Reddit API Credentials (optional)

Fuer Reddit-Ingestion werden API-Zugangsdaten benoetigt:

1. https://www.reddit.com/prefs/apps aufrufen
2. "Create App" -> "script" waehlen
3. Client ID und Secret in `.env` eintragen:

```env
REDDIT_CLIENT_ID=dein_client_id
REDDIT_CLIENT_SECRET=dein_client_secret
REDDIT_USER_AGENT=rag-bot/0.1
```

### Schritt 5: Twitter/X Session (optional)

Twikit benoetigt Session-Cookies von einem eingeloggten X-Account:

```python
from twikit import Client
client = Client("de-DE")
await client.login(
    auth_info_1="dein_username",
    auth_info_2="deine_email@example.com",
    password="dein_passwort"
)
client.save_cookies("twitter_cookies.json")
```

Dann beim Erstellen des TwitterIngestor den Pfad angeben:

```python
ingestor = TwitterIngestor(cookies_path="twitter_cookies.json")
```

### Schritt 6: End-to-End Test mit echten Daten [ERLEDIGT]

```bash
cd /root/rag
source .venv/bin/activate

# 1. Ein PDF ingesten
python -m rag.cli ingest /pfad/zu/dokument.pdf

# 2. Eine Website ingesten
python -m rag.cli ingest "https://de.wikipedia.org/wiki/Berlin"

# 3. Ein YouTube Video ingesten
python -m rag.cli ingest "https://www.youtube.com/watch?v=VIDEO_ID"

# 4. Suchen
python -m rag.cli search "Was ist die Hauptstadt von Deutschland?"

# 5. Frage stellen (benoetigt LLM auf Mac Studio)
python -m rag.cli ask "Was weisst du ueber Berlin?"

# 6. Statistiken
python -m rag.cli stats
```

### Schritt 7: FastAPI Server starten [ERLEDIGT]

```bash
cd /root/rag
source .venv/bin/activate

# API auf Port 8000 starten (erreichbar im LAN)
uvicorn rag.api:app --host 0.0.0.0 --port 8000

# Oder in tmux fuer Dauerbetrieb:
tmux new -s api
uvicorn rag.api:app --host 0.0.0.0 --port 8000
# Ctrl+B, D zum Detachen
```

**API Endpoints:**

| Method | Endpoint | Beschreibung |
|--------|----------|-------------|
| GET | `/health` | Healthcheck |
| POST | `/ingest` | Dokument ingesten (`{"source": "...", "type": "auto"}`) |
| GET | `/search?q=...` | Hybrid Search (optional: `&platform=pdf&author=...`) |
| POST | `/ask` | Frage mit Citations (`{"question": "...", "platform": null}`) |
| GET | `/stats` | Systemstatistiken |

### Schritt 8: Prefect Pipeline fuer taegliche Ingestion [ERLEDIGT]

Prefect 3.6 installiert. Taeglich um 06:00 Uhr laeuft der `daily-ingestion` Flow.

**Quellen (konfiguriert in `sources.yaml`):**

- **Web:** Feste URL-Liste
- **YouTube:** Watch Later Playlist + gespeicherte Playlists (benoetigt OAuth)
- **Reddit:** Gespeicherte Posts des Users (benoetigt Credentials)
- **Twitter/X:** Gebookmarkte Tweets (benoetigt Cookies)

**Dedup:** URL-Check gegen PostgreSQL -- bereits ingestierte URLs werden uebersprungen.

**Graceful Skip:** Fehlende Credentials fuehren nicht zu Fehlern, die Quelle wird uebersprungen.

**Module:**

```
src/rag/pipeline/
  sources.py    # YAML-Parser, Dataclasses (WebSource, YouTubeConfig, etc.)
  dedup.py      # URL-Dedup via PostgreSQL
  tasks.py      # Prefect Tasks (ingest_web_url, fetch_reddit_saved, etc.)
  flows.py      # Haupt-Flow daily_ingestion
  deploy.py     # Deployment mit Cron-Schedule
```

**Starten:**

```bash
# Prefect Server (Web-UI auf Port 4200)
tmux new -s prefect
export PATH=/root/.local/bin:$PATH
source .venv/bin/activate
prefect server start --host 0.0.0.0

# Pipeline Worker (fuehrt Flows aus)
tmux new -s pipeline
export PATH=/root/.local/bin:$PATH
source .venv/bin/activate
PREFECT_API_URL=http://localhost:4200/api python -m rag.pipeline.deploy
```

**Manueller Flow-Run:**

```bash
PREFECT_API_URL=http://localhost:4200/api prefect deployment run daily-ingestion/daily-ingestion
```

**Web-UI:** http://192.168.178.182:4200

**Noch offen fuer volle Funktionalitaet:**

- YouTube: `google-api-python-client` installieren + OAuth Credentials
- Reddit: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, `REDDIT_PASSWORD` in .env
- Twitter: Cookies-Datei unter `/root/rag/twitter_cookies.json`

---

## Betrieb

### Services starten/stoppen

```bash
# Docker Services (Qdrant, Neo4j, PostgreSQL)
cd /root/rag
docker compose up -d      # starten
docker compose ps          # Status
docker compose logs -f     # Logs
docker compose down        # stoppen (Daten bleiben erhalten)

# Alles komplett loeschen (ACHTUNG: Datenverlust!)
# docker compose down -v
```

### tmux Sessions

**Laufende Sessions:**

| Session | Dienst | Port |
|---------|--------|------|
| `api` | FastAPI Server | 8000 |
| `prefect` | Prefect Server (Web-UI) | 4200 |
| `pipeline` | Pipeline Worker (Cron) | -- |

```bash
# Alle Sessions anzeigen
tmux ls

# Zu einer Session wechseln
tmux attach -t api

# Session verlassen (laeuft weiter)
# Ctrl+B, dann D

# Neue Session starten
tmux new -s <name>
```

### Claude Code auf dem Container

```bash
ssh root@192.168.178.182
tmux new -s claude
cd /root/rag
claude
```

### Backups

Die Docker Volumes liegen unter `/var/lib/docker/volumes/`:
- `rag_qdrant_data` -- Alle Vektoren und Payloads
- `rag_neo4j_data` -- Knowledge Graph
- `rag_postgres_data` -- Dokument-Metadaten

```bash
# Einfaches Backup der Volumes
docker compose stop
tar czf /root/rag-backup-$(date +%Y%m%d).tar.gz \
  /var/lib/docker/volumes/rag_qdrant_data \
  /var/lib/docker/volumes/rag_neo4j_data \
  /var/lib/docker/volumes/rag_postgres_data
docker compose up -d
```

### Tests ausfuehren

```bash
cd /root/rag
source .venv/bin/activate

# Alle Tests (ohne Netzwerk-Tests)
pytest -v -m "not network"

# Nur Storage-Tests
pytest tests/storage/ -v

# Nur NER-Tests
pytest tests/processing/test_ner.py -v

# Mit Netzwerk-Tests (braucht Internet + API-Keys)
pytest -v
```

### Monitoring

```bash
# RAM-Nutzung
free -h

# Docker Container Status
docker stats --no-stream

# Disk Usage
df -h /
du -sh /var/lib/docker/volumes/rag_*

# Qdrant Healthcheck
curl -s localhost:6333/healthz

# Neo4j Status
curl -s localhost:7474

# PostgreSQL
pg_isready -h localhost -p 5432 -U rag
```

---

## Bekannte Einschraenkungen

1. **Embedding auf CPU:** BGE-M3 laeuft auf CPU im LXC (~3 Sek/Text). Fuer Bulk-Ingestion
   von >10.000 Dokumenten empfiehlt sich Batch-Processing ueber Nacht.

2. **Twitter/X fragil:** Twikit nutzt Reverse-Engineering der X-API. Kann jederzeit brechen.
   Gutes Error-Handling ist implementiert, aber Ausfaelle sind zu erwarten.

3. **LLM-Abhaengigkeit:** Frage-Antwort (`ask`) benoetigt laufendes LLM (MiniMax M2.5)
   auf dem Mac Studio (.8:54321). Ingestion und Suche funktionieren unabhaengig davon.

4. **Keine GPU im LXC:** NER und Topic Modeling laufen auf CPU. Fuer den geplanten
   Dokumentenumfang (50-100K) ist das ausreichend, dauert aber bei Erstverarbeitung.

5. **Kein HTTPS:** API und Neo4j Browser laufen ohne TLS. Nur fuer LAN-Nutzung geeignet.

---

## Schnellreferenz

```bash
# SSH zum Container
ssh root@192.168.178.182

# Projekt aktivieren
cd /root/rag && source .venv/bin/activate
export PATH=/root/.local/bin:$PATH

# CLI Befehle
python -m rag.cli ingest <quelle>        # PDF/URL/YouTube ingesten
python -m rag.cli search "suchbegriff"    # Hybrid-Suche
python -m rag.cli ask "frage"             # Q&A mit Quellennachweisen
python -m rag.cli stats                   # Systemstatistiken

# API starten
uvicorn rag.api:app --host 0.0.0.0 --port 8000

# Prefect
prefect server start --host 0.0.0.0      # Server (Port 4200)
PREFECT_API_URL=http://localhost:4200/api python -m rag.pipeline.deploy  # Worker

# Tests
pytest -v -m "not network"

# Docker Services
docker compose ps                         # Status
docker compose up -d                      # Starten
docker compose logs -f                    # Logs
```
