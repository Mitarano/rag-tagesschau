import time, json, re, sqlite3, feedparser, httpx
from selectolax.parser import HTMLParser

FEED = "https://www.tagesschau.de/infoservices/alle-meldungen-100~rss2.xml"
UA = "WorkshopCrawler"
DB = "tagesschau.sqlite"

def init_db():
    con = sqlite3.connect(DB)
    con.execute("""\
        CREATE TABLE IF NOT EXISTS docs (
          guid TEXT PRIMARY KEY,
          url TEXT,
          title TEXT,
          published TEXT,
          modified TEXT,
          text TEXT,
          fetched_at INTEGER)""")
    con.commit()
    con.close()


def save_doc(doc):
    con = sqlite3.connect(DB)
    con.execute("INSERT OR REPLACE INTO docs (guid,url,title,published,modified,text,fetched_at) VALUES (?,?,?,?,?,?,?)",
        (doc["guid"], doc["url"], doc["title"], doc["published"], doc["modified"], doc["text"], int(time.time())))
    con.commit()
    con.close()


def pick_article_fields(o):
    if not isinstance(o, dict):
        return None
    
    t = o.get("@type")

    if isinstance(t, list):
        t = next((x for x in t if isinstance(x, str) and ("Article" in x)), None)

    if not (isinstance(t, str) and ("Article" in t)):
        return None
    
    title = o.get("headline")
    text  = o.get("articleBody")
    datePublished  = o.get("datePublished")
    dateModified = o.get("dateModified")

    if isinstance(text, list):
        text = " ".join([x for x in text if isinstance(x, str)])

    title = re.sub(r"\s+", " ", title).strip() if title else None
    text  = re.sub(r"\s+", " ", text).strip()
    
    if text:
        return {
            "title": title,
            "text": text,
            "published": datePublished,
            "modified": dateModified
        } 
    else:
        return None


def extract_jsonld_article(html):
    doc = HTMLParser(html)

    for node in doc.css('script[type="application/ld+json"]'):
        try:
            data = json.loads(node.text())
        except Exception:
            continue

        items = data if isinstance(data, list) else [data]

        for obj in items:
            art = pick_article_fields(obj)
            if art:
                return art
            
    return None


def fetch(url):
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=20) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.text


def handle_item(e):
    guid = e.get("guid")
    url  = e.link
    html = fetch(url)
    art  = extract_jsonld_article(html)
    
    return {
        "guid": guid,
        "url": url,
        "title": art.get("title") or e.get("title"),
        "published": art.get("published") or e.get("published"),
        "modified": art.get("modified") or e.get("modified"),
        "text": art["text"]
    }


def main():
    init_db()
    feed = feedparser.parse(FEED)
    count = 0

    for e in feed.entries:
        try:
            doc = handle_item(e)

            if doc:
                save_doc(doc)
                count += 1

            time.sleep(0.5)

        except Exception as ex:
            print("Fehler bei:", e.get("link"))
            print("Fehlerdetails:", ex)
            continue

    print(f"gespeichert: {count} Artikel")


if __name__ == "__main__":
    main()
