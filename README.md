# RAG mit Artikeln von der Tagesschau

## Voraussetzungen
- Python 3.10+
- [Ollama](https://ollama.com/) installiert und gestartet (`ollama serve`)
- Modelle (einmalig laden):
  ```bash
  ollama pull bge-m3      # Embedding
  ollama pull qwen3:8b    # Chat (alternativ: llama3:8b)
  ```
- Python-Pakete:
  ```bash
  pip install feedparser httpx selectolax chromadb ollama
  ```

---

## Skripte

- **`crawler.py`**  
  Holt Artikel aus dem Tagesschau-RSS, extrahiert Volltext via JSON-LD und speichert nach **SQLite**.

- **`indexer.py`**  
  Liest aus SQLite, berechnet **Embeddings mit Ollama** und schreibt/aktualisiert eine **Chroma-Collection**.

- **`chat.py`**  
  Simple **RAG-Abfrage**: Query → Embedding → Dokumente suchen → Antwort.

---

## Nutzung

1) **Crawler ausführen** (Daten einsammeln / aktualisieren)
   ```bash
   python crawler.py
   ```

2) **Index aufbauen/aktualisieren** (Embeddings → Chroma)
   ```bash
   python indexer.py
   ```

3) **Fragen stellen** (RAG)
   ```bash
   python chat.py "Was wurde zuletzt zur Energiepolitik berichtet?"
   # optional:
   python chat.py --stream -k 6 "Deine Frage"
   ```

> Für Updates: Schritt **1** (neue Artikel) und danach **2** (Index aktualisieren).  
> Zum reinen Abfragen genügt **3** – der Index bleibt erhalten.
