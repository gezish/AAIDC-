# ingest/build_index.py
import os, json, typer, yaml
from typing import List, Optional
from loguru import logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from utils.config import load_config
from ingest.clean import normalize_for_dedup
from ingest.loaders import (
    load_publications_from_dir,
    load_publications_from_json,
    load_docs_from_urls,
    load_wikipedia_pages,
)

app = typer.Typer(help="Build FAISS indexes for datasets (local, free stack).")

def get_splitter(dataset: str):
    if dataset == "docs":
        return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120, separators=["\n# ", "\n## ", "\n", " "])
    elif dataset == "wikipedia":
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    # publications default
    return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

def deduplicate_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        key = (normalize_for_dedup(d.page_content), d.metadata.get("publication_id") or d.metadata.get("source") or d.metadata.get("source_url") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out

def make_embeddings(model_name: str, normalize: bool = True):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": normalize},
    )

def persist_metadata(chunks: List[Document], outpath: str):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        for d in chunks:
            rec = {"text": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved chunk metadata to {outpath}")

def load_field_map(path: Optional[str]):
    if not path:
        return None
    if path.lower().endswith((".yml", ".yaml")):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.command()
def build(
    dataset: str = typer.Option(..., help="publications | docs | wikipedia"),
    config_path: str = typer.Option("config/config.yaml", help="Path to config"),
    # Publications inputs
    raw_dir: str = typer.Option("data/raw/publications", help="Dir for PDFs/MD/TXT (publications)"),
    pub_json: Optional[str] = typer.Option(None, help="Path to publications JSON (.json/.jsonl) or directory"),
    field_map: Optional[str] = typer.Option(None, help="Optional field map file (.yaml/.json) for publications"),
    # Docs/Wiki inputs
    docs_urls: str = typer.Option("data/sources/docs_urls.txt", help="Docs URLs file (docs)"),
    wiki_titles: str = typer.Option("data/sources/wikipedia_titles.txt", help="Wikipedia titles file (wikipedia)"),
    wiki_lang: str = typer.Option("en", help="Wikipedia language code"),
    save_chunks: bool = typer.Option(True, help="Save processed chunks to JSONL"),
):
    cfg = load_config(config_path, dataset)
    logger.info(f"Loaded config for dataset={dataset}")

    # 1) Load
    docs: List[Document] = []
    if dataset == "publications":
        if pub_json:
            fmap = load_field_map(field_map)
            docs = load_publications_from_json(pub_json, field_map=fmap)
            logger.info(f"Loaded publications from JSON path={pub_json}")
        else:
            docs = load_publications_from_dir(raw_dir)
            logger.info(f"Loaded publications from raw_dir={raw_dir}")
    elif dataset == "docs":
        docs = load_docs_from_urls(docs_urls)
    elif dataset == "wikipedia":
        docs = load_wikipedia_pages(wiki_titles, lang=wiki_lang)
    else:
        raise typer.BadParameter("dataset must be publications | docs | wikipedia")

    logger.info(f"Loaded {len(docs)} raw docs/sections")

    # 2) Deduplicate
    docs = deduplicate_docs(docs)
    logger.info(f"{len(docs)} docs/sections after dedup")

    # 3) Chunk
    splitter = get_splitter(dataset)
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.split()) > 30]
    logger.info(f"Created {len(chunks)} chunks")

    # 4) Embed + index
    emb_model = cfg["embeddings"]["model_name"]
    normalize = bool(cfg["embeddings"].get("normalize", True))
    embedder = make_embeddings(emb_model, normalize=normalize)

    index_dir = cfg["vector_store"]["path"]
    out_dir = os.path.join(index_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    vs = FAISS.from_documents(chunks, embedder)
    vs.save_local(out_dir)
    logger.info(f"FAISS index saved to {out_dir}")

    # 5) Save processed chunks
    if save_chunks:
        chunks_out = f"data/processed/{dataset}/chunks.jsonl"
        persist_metadata(chunks, chunks_out)

    logger.success(f"Ingestion complete for dataset={dataset}")

if __name__ == "__main__":
    app()