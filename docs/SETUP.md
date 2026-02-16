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
  Ollama API :11434
    Qwen 2.5-72B Q8     (~75GB)   Kern-RAG
    MiniMax M2.5 Q4      (~138GB)  Agent/Router
    Qwen3-Reranker-8B    (~10GB)   Reranking
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
    reranker.py      # Qwen3-Reranker via Ollama API
  generation/
    llm.py           # Ollama REST Client
    router.py        # Query Router (einfach->Qwen, komplex->MiniMax)
    citation.py      # Citation-aware Antworten mit [1], [2] Referenzen
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

### Schritt 1: Ollama auf dem Mac Studio einrichten

Voraussetzung: Ollama muss installiert sein. Falls nicht:

```bash
# Auf dem Mac Studio
brew install ollama
```

**1a. Ollama so konfigurieren, dass es im LAN erreichbar ist:**

```bash
# Auf dem Mac Studio
# Ollama muss auf 0.0.0.0 lauschen statt nur localhost
launchctl setenv OLLAMA_HOST 0.0.0.0

# Ollama neu starten
brew services restart ollama
# oder: ollama serve
```

Verifizieren:

```bash
# Vom LXC Container aus
curl http://<MAC-STUDIO-IP>:11434/api/tags
```

**1b. Modelle herunterladen:**

```bash
# Auf dem Mac Studio
# Kern-RAG Modell (~75GB, Q8 Quantisierung)
ollama pull qwen2.5:72b-instruct-q8_0

# Reranker (~10GB)
ollama pull qwen3-reranker:8b

# MiniMax laeuft bereits - pruefen:
ollama list | grep minimax
```

Hinweis: Der Download von Qwen 2.5-72B dauert je nach Internetverbindung
30-60 Minuten. Das Modell belegt ~75GB RAM beim Laden.

**1c. RAM-Budget pruefen:**

| Modell | RAM |
|--------|-----|
| Qwen 2.5-72B Q8 | ~75 GB |
| MiniMax M2.5 Q4 | ~138 GB |
| Qwen3-Reranker-8B | ~10 GB |
| **Gesamt (alle gleichzeitig)** | **~223 GB** |
| **Verfuegbar (Mac Studio 512GB)** | **~300 GB frei** |

Ollama laedt Modelle bei Bedarf in den RAM und entlaedt sie nach Inaktivitaet.
Im Normalbetrieb sind nicht alle Modelle gleichzeitig geladen.

### Schritt 2: .env Datei konfigurieren

```bash
# Auf dem LXC Container
ssh root@192.168.178.182
cd /root/rag

# .env aus .env.example erstellen
cp .env.example .env

# Die Mac Studio IP eintragen
nano .env
```

**Mindestens diese Werte anpassen:**

```env
# Die tatsaechliche IP des Mac Studio eintragen
OLLAMA_HOST=192.168.178.XXX

# Falls die Ollama-Modellnamen anders sind, anpassen:
OLLAMA_MODEL_RAG=qwen2.5:72b-instruct-q8_0
OLLAMA_MODEL_AGENT=minimax-m2.5:q4_K_M
OLLAMA_MODEL_RERANKER=qwen3-reranker:8b
```

### Schritt 3: Netzwerk-Konnektivitaet testen

```bash
# Auf dem LXC Container

# Ollama API erreichbar?
curl -s http://<MAC-STUDIO-IP>:11434/api/tags | python3 -m json.tool

# Modell-Test (sollte eine Antwort generieren)
curl -s http://<MAC-STUDIO-IP>:11434/api/generate \
  -d '{"model": "qwen2.5:72b-instruct-q8_0", "prompt": "Hallo, antworte kurz.", "stream": false}' \
  | python3 -m json.tool

# Automatischer Test via RAG-Code
cd /root/rag
source .venv/bin/activate
python -c "
from rag.generation.llm import OllamaClient
client = OllamaClient()
print('Ollama erreichbar:', client.is_available())
print('Qwen verfuegbar:', client.is_available('qwen2.5'))
print('MiniMax verfuegbar:', client.is_available('minimax'))
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

### Schritt 6: End-to-End Test mit echten Daten

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

# 5. Frage stellen (benoetigt Ollama-Verbindung)
python -m rag.cli ask "Was weisst du ueber Berlin?"

# 6. Statistiken
python -m rag.cli stats
```

### Schritt 7: FastAPI Server starten (optional)

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

### Schritt 8: Prefect Pipeline fuer taegliche Ingestion (noch nicht implementiert)

Wenn gewuenscht, wird ein Prefect Flow erstellt der:
- Konfigurierte Quellen taeglich abfragt (YouTube-Kanaele, Subreddits, Webseiten)
- Neue Dokumente automatisch ingestiert
- Fehler protokolliert, ohne andere Quellen zu blockieren

```bash
# Prefect installieren
cd /root/rag
source /root/.local/bin/env
uv pip install prefect
```

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

```bash
# Neue Session starten
tmux new -s rag

# Session verlassen (laeuft weiter)
# Ctrl+B, dann D

# Zurueck zur Session
tmux attach -t rag

# Alle Sessions anzeigen
tmux ls
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

3. **Ollama-Abhaengigkeit:** Frage-Antwort (`ask`) und Reranking benoetigen laufendes Ollama
   auf dem Mac Studio. Ingestion und Suche funktionieren unabhaengig davon.

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

# CLI Befehle
python -m rag.cli ingest <quelle>        # PDF/URL/YouTube ingesten
python -m rag.cli search "suchbegriff"    # Hybrid-Suche
python -m rag.cli ask "frage"             # Q&A mit Quellennachweisen
python -m rag.cli stats                   # Systemstatistiken

# API starten
uvicorn rag.api:app --host 0.0.0.0 --port 8000

# Tests
pytest -v -m "not network"

# Docker Services
docker compose ps                         # Status
docker compose up -d                      # Starten
docker compose logs -f                    # Logs
```
