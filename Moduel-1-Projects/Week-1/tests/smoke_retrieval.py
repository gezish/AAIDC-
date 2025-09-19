# tests/smoke_retrieval.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

dataset = "publications"
vs_dir = f"data/indexes/{dataset}"

embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

db = FAISS.load_local(vs_dir, embedder, allow_dangerous_deserialization=True)

queries = [
    "What's this publication about?",
    "What models or tools were used?",
    "Any limitations or assumptions?",
]
for q in queries:
    print("\nQ:", q)
    docs = db.similarity_search(q, k=5)
    for i, d in enumerate(docs, 1):
        print(f" {i}. {d.metadata.get('title')} — {d.metadata.get('section')}")
        print("    src:", d.metadata.get("source_url") or d.metadata.get("source"))
        print("    snippet:", d.page_content[:160].replace("\n", " "), "...")