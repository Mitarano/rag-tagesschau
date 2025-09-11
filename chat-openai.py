import os
import sys
import argparse
import chromadb, ollama
from dotenv import load_dotenv
from openai import OpenAI

CHROMA_DIR = "chroma_tagesschau"
COLL_NAME  = "tagesschau"
CHAT_MODEL = "gpt-4.1"
EMB_MODEL  = "bge-m3"

load_dotenv(override=True)
oai = OpenAI()

client = chromadb.PersistentClient(path=CHROMA_DIR)
coll = client.get_or_create_collection(COLL_NAME)


def embed(txt: str):
    return ollama.embeddings(model=EMB_MODEL, prompt=txt)["embedding"]


def retrieve(query: str, k: int = 4):
    """Hole relevante Dokumente aus Chroma (früher: search)."""
    qv = embed(query)
    res = coll.query(query_embeddings=[qv], n_results=k)
    hits = []

    if res["ids"]:
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({"doc": doc, "meta": meta, "distance": float(dist)})

    return hits


def _build_context(hits):
    ctx = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        ctx.append(
            f"=== DOC {i} ===\n"
            f"Titel: {m.get('title','')}\n"
            f"Datum: {m.get('date','')}\n"
            f"URL:   {m.get('url','')}\n"
            f"Text:\n{h['doc']}\n"
            f"=== END DOC {i} ==="
        )
    return "\n\n".join(ctx)


def generate(query: str, hits, stream: bool = False):
    if not hits:
        return "Keine passenden Artikel."

    context = _build_context(hits)
    
    system = (
        "Antworte ausschließlich auf Basis der mitgegebenen Dokumente. "
        "Wenn die Frage nicht auf Basis dieser Dokumente beantwortet werden kann, "
        "sage dies ausdrücklich."
    )
    
    user = (
        f"Dokumente:\n{context}\n\n"
        f"Frage: {query}\n\n"
        f"Antworte kurz (3-5 Sätze) und nenne die Quellen (Titel und vollständige URL)."
    )
    inputs = [
        {"role": "system", "content": [{"type": "input_text", "text": system}]},
        {"role": "user",   "content": [{"type": "input_text", "text": user}]},
    ]

    if stream:
        with oai.responses.stream(model=CHAT_MODEL, input=inputs) as s:
            try:
                for event in s:
                    if event.type == "response.output_text.delta":
                        print(event.delta, end="", flush=True)

                print()

            except KeyboardInterrupt:
                print("\n[abgebrochen]")

        return None

    resp = oai.responses.create(model=CHAT_MODEL, input=inputs)
    print(resp.output_text)


def answer(query: str, k: int = 4):
    hits = retrieve(query, k=k)

    if not hits:
        return "Keine passenden Artikel."
    
    generate(query, hits, stream=False)


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Simple RAG mit retrieve + generate (optional Streaming).")
    p.add_argument("prompt", nargs="?", help="Frage/Prompt für das RAG.")
    p.add_argument("-k", type=int, default=4, help="Anzahl der zu holenden Treffer (default: 4).")
    p.add_argument("--stream", action="store_true", help="Antwort wird gestreamt.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    query = args.prompt if args.prompt is not None else input("Frage: ").strip()

    hits = retrieve(query, k=args.k)

    if not hits:
        print("Keine passenden Artikel.")
        sys.exit(0)

    result = generate(query, hits, stream=args.stream)
