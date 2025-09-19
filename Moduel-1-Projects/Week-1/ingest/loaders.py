
import os, json, hashlib, time, requests
from typing import Dict, List, Iterable, Any, Optional
import trafilatura
import wikipediaapi
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from ingest.clean import basic_clean, normalize_for_dedup, segment_by_divider_and_headings
from ingest.extractors import extract_publication_fields

def _hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def _ensure_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        if x and isinstance(x[0], dict) and "name" in x[0]:
            return [d.get("name") for d in x if d.get("name")]
        return [str(v) for v in x]
    return [str(x)]

DEFAULT_FIELD_MAP: Dict[str, List[str]] = {
    "id": ["id", "paper_id", "uuid"],
    "username": ["username", "author", "owner"],
    "title": ["title", "name"],
    "authors": ["authors", "author_list", "creators"],
    "date": ["date", "published_at", "publication_date", "year"],
    "url": ["url", "source_url", "link"],
    "license": ["license"],
    # Content fields
    "description": ["publication_description", "description"],
    "abstract": ["abstract", "summary"],
    "body": ["body", "content", "full_text", "text"],
    "sections": ["sections", "section_list"],
    "section_title": ["heading", "title", "name"],
    "section_text": ["text", "content", "body"],
    # Domain fields (if present)
    "models_used": ["models_used", "models"],
    "tools_used": ["tools_used", "tools"],
    "limitations": ["limitations"],
    "assumptions": ["assumptions"]
}

def _get_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default

def _iter_json_records(path: str) -> Iterable[Dict[str, Any]]:
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                full = os.path.join(root, f)
                if f.lower().endswith((".jsonl", ".ndjson")):
                    with open(full, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                yield json.loads(line)
                            except Exception:
                                continue
                elif f.lower().endswith(".json"):
                    try:
                        with open(full, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                            if isinstance(data, list):
                                for rec in data:
                                    if isinstance(rec, dict):
                                        yield rec
                            elif isinstance(data, dict):
                                yield data
                    except Exception:
                        continue
    else:
        if path.lower().endswith((".jsonl", ".ndjson")):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
        else:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        yield rec
            elif isinstance(data, dict):
                yield data

def load_publications_from_json(input_path: str, field_map: Optional[Dict[str, Any]] = None) -> List[Document]:
    fmap = {}
    user_map = field_map or {}
    for k, defaults in DEFAULT_FIELD_MAP.items():
        v = user_map.get(k, defaults)
        fmap[k] = v if isinstance(v, list) else [v]

    docs: List[Document] = []
    for rec in _iter_json_records(input_path):
        pub_id = _get_first(rec, fmap["id"]) or _hash_id(json.dumps(rec, ensure_ascii=False)[:2000])
        title = _get_first(rec, fmap["title"], "Untitled")
        authors = _ensure_list(_get_first(rec, fmap["authors"]))
        username = _get_first(rec, fmap["username"])
        if not authors and username:
            authors = [str(username)]
        date = _get_first(rec, fmap["date"])
        url = _get_first(rec, fmap["url"])
        license_ = _get_first(rec, fmap["license"], "unknown")

        # Content sources
        description = _get_first(rec, fmap["description"])
        abstract = _get_first(rec, fmap["abstract"])
        body = _get_first(rec, fmap["body"])
        sections = _get_first(rec, fmap["sections"], [])

        base_meta = {
            "publication_id": pub_id,
            "title": title,
            "authors": authors,
            "date": date,
            "source": input_path,
            "source_url": url,
            "source_type": "json",
            "doc_type": "publication",
            "license": license_,
            "username": username,
        }

        # Use description with DIVIDERs if present; else structured sections; else abstract/body
        section_docs: List[Document] = []
        if isinstance(description, str) and description.strip():
            parsed = segment_by_divider_and_headings(description)
            # One-time extraction across the whole description
            fields = extract_publication_fields(description)
            base_meta_with_fields = {**base_meta, **fields}
            for section_title, content in parsed:
                meta = {**base_meta_with_fields, "section": section_title}
                section_docs.append(Document(page_content=content, metadata=meta))
        elif isinstance(sections, list) and sections:
            for sec in sections:
                sec_title = _get_first(sec, fmap["section_title"], "Section")
                sec_text = _get_first(sec, fmap["section_text"])
                if not sec_text:
                    continue
                content = basic_clean(sec_text)
                if len(content.split()) < 5:
                    continue
                meta = {**base_meta, "section": sec_title}
                section_docs.append(Document(page_content=content, metadata=meta))
        else:
            # Fallback: abstract + body
            if abstract:
                section_docs.append(Document(page_content=basic_clean(abstract), metadata={**base_meta, "section": "Abstract"}))
            if body:
                section_docs.append(Document(page_content=basic_clean(body), metadata={**base_meta, "section": "Body"}))

        docs.extend(section_docs)

    return docs

# Optional loaders for other datasets (kept for completeness)
def load_publications_from_dir(raw_dir: str) -> List[Document]:
    docs: List[Document] = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            path = os.path.join(root, f)
            lower = f.lower()
            if lower.endswith(".pdf"):
                for d in PyPDFLoader(path).load():
                    content = basic_clean(d.page_content)
                    if not content:
                        continue
                    meta = d.metadata or {}
                    meta.update({"source": path, "source_type": "pdf", "title": meta.get("title") or os.path.basename(path)})
                    docs.append(Document(page_content=content, metadata=meta))
            elif lower.endswith((".md", ".txt")):
                for d in TextLoader(path, encoding="utf-8").load():
                    content = basic_clean(d.page_content)
                    if not content:
                        continue
                    meta = d.metadata or {}
                    meta.update({"source": path, "source_type": "text", "title": meta.get("title") or os.path.basename(path)})
                    docs.append(Document(page_content=content, metadata=meta))
    return docs

def load_docs_from_urls(urls_file: str) -> List[Document]:
    docs: List[Document] = []
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip() and not u.strip().startswith("#")]
    for url in urls:
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                continue
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
            if not extracted:
                html = requests.get(url, timeout=20).text
                soup = BeautifulSoup(html, "lxml")
                extracted = soup.get_text("\n")
            content = basic_clean(extracted)
            if not content:
                continue
            title = url.split("/")[-1].replace("-", " ").title()
            meta = {"source_url": url, "title": title, "source_type": "web"}
            docs.append(Document(page_content=content, metadata=meta))
        except Exception:
            continue
        time.sleep(0.2)
    return docs

def load_wikipedia_pages(titles_file: str, lang: str = "en") -> List[Document]:
    wiki = wikipediaapi.Wikipedia(language=lang)
    docs: List[Document] = []
    with open(titles_file, "r", encoding="utf-8") as f:
        titles = [t.strip() for t in f if t.strip() and not t.strip().startswith("#")]
    for title in titles:
        page = wiki.page(title)
        if not page.exists():
            continue
        content = basic_clean(page.text)
        meta = {
            "title": page.title,
            "source_url": f"https://{lang}.wikipedia.org/wiki/{page.title.replace(' ', '_')}",
            "license": "CC BY-SA 4.0",
            "source_type": "wikipedia",
            "doc_type": "wikipedia"
        }
        docs.append(Document(page_content=content, metadata=meta))
    return docs