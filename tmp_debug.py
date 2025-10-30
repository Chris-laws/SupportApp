import numpy as np
from pathlib import Path
from app.chatbot.modules.embeddings import load_faiss_index, get_embeddings
from app.chatbot.modules.retriever import HybridRetriever

index, records, embeddings = load_faiss_index("app/chatbot/data/faiss_index/index")
retriever = HybridRetriever(records, embeddings=embeddings, faiss_index=index)
query = "Softphone Fusion Leiste Funktionen"
vec = get_embeddings([query])[0]
ranked = retriever.retrieve(query, np.array(vec, dtype=np.float32), top_k=150, bm25_k=600, reranker=None, mode="hybrid")
for rank, item in enumerate(ranked[:10], start=1):
    doc = Path(item.get("source") or "").name
    page = item.get("page")
    print(rank, doc, page, item.get("keyword_coverage"), round(item.get("score",0),3))
