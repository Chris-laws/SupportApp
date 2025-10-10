# Support Assistant App

Dieser Ordner enthält die FastAPI-basierte RAG-Anwendung inkl. aller benötigten Assets.

## Projektstruktur

```
SupportAssistantApp/
├─ app/
│  └─ chatbot/
│     ├─ webserver.py        # FastAPI Einstiegspunkt
│     ├─ modules/            # Embeddings, Retriever, LLM-Anbindung
│     ├─ templates/          # Jinja2-Frontends (index.html)
│     ├─ data/
│     │  ├─ *.pdf            # Knowledge Base Quellen
│     │  └─ faiss_index/     # Index + Metadaten
│     └─ prompts/            # Prompt-Vorlagen
├─ docs/                     # Systemarchitektur (Mermaid + Exporte)
├─ scripts/experiments/      # Hilfsskripte und Tests
├─ node_modules/             # Mermaid CLI Abhängigkeiten
├─ package.json / lock       # Toolkit-Konfiguration (Mermaid CLI)
```

## Anwendung starten

```powershell
cd C:\Users\IT-Admin\SupportAssistantApp
$env:PYTHONPATH = "app\chatbot"
uvicorn webserver:app --reload --host 0.0.0.0 --port 8000
```

Standardendpunkte:
- `GET /` – HTML-Frontend
- `POST /ask` – JSON API
- `POST /ask-html` – Form Endpoint

## Diagramm aktualisieren

```powershell
cd C:\Users\IT-Admin\SupportAssistantApp
npx mmdc -i docs/system-architecture.mmd -o docs/system-architecture.png
npx mmdc -i docs/system-architecture.mmd -o docs/system-architecture.pdf
```

## Hinweise
- Projektname wurde in `SupportAssistantApp` geändert; frühere Pfade `RagChatbotFront` sind nicht mehr nötig.
- Falls du git nutzt, Repository erst initialisieren (`git init`) oder neu klonen.
- `node_modules` ist groß – je nach VCS in `.gitignore` aufnehmen.
