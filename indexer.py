import sqlite3, hashlib
import chromadb
from chromadb.config import Settings
import ollama

DB = "tagesschau.sqlite"
CHROMA_DIR = "chroma_tagesschau"
COLL_NAME  = "tagesschau"
EMB_MODEL  = "bge-m3"

client = chromadb.PersistentClient(path=CHROMA_DIR)
coll = client.get_or_create_collection(COLL_NAME)


def embed(txt: str):
    return ollama.embeddings(model=EMB_MODEL, prompt=txt)["embedding"]


def content_hash(title, text):
    h = hashlib.sha256()
    h.update(((title or "")+"\n\n"+(text or "")).encode("utf-8", "ignore"))
    return h.hexdigest()


def fetch_docs(limit=None):
    con = sqlite3.connect(DB)
    sql = "SELECT guid, url, title, published, modified, text FROM docs ORDER BY fetched_at DESC"
    if limit: sql += f" LIMIT {int(limit)}"
    rows = con.execute(sql).fetchall()
    con.close()
    return rows


def upsert_all(limit=None):
    rows = fetch_docs(limit)
    new, skipped = 0, 0

    for guid, url, title, pub, mod, text in rows:
        if not guid or not text: 
            continue

        h = content_hash(title, text)
        existing = coll.get(ids=[guid], include=["metadatas"])

        if existing["ids"]:
            meta = existing["metadatas"][0] or {}
            if meta.get("hash") == h:
                skipped += 1
                continue

        vec = embed(((title or "")+"\n\n"+text).strip())

        coll.upsert(
            ids=[guid],
            documents=[text],
            metadatas=[{
                "title": title,
                "url": url,
                "published": pub,
                "modified": mod,
                "hash": h
            }],
            embeddings=[vec]
        )
        new += 1

    print(f"Chroma: {new} neu/aktualisiert, {skipped} übersprungen.")


if __name__ == "__main__":
    upsert_all()
