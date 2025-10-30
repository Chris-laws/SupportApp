import numpy as np
from pathlib import Path
import json
from app.chatbot.modules.embeddings import load_faiss_index, get_embeddings
from app.chatbot.modules.retriever import HybridRetriever
from app.chatbot.modules.reranker import CrossEncoderReranker

index, records, embeddings = load_faiss_index("app/chatbot/data/faiss_index/index")
retriever = HybridRetriever(records, embeddings=embeddings, faiss_index=index)
reranker = CrossEncoderReranker()

qs = json.loads(Path("data/eval/questions_gold.json").read_text(encoding="utf-8"))
more = [sample for sample in qs if sample["question_id"] >= "G016"]
for sample in more:
    qid = sample["question_id"]
    query = sample["question"]
    vec = get_embeddings([query])[0]
    ranked = retriever.retrieve(query, np.array(vec, dtype=np.float32), top_k=80, bm25_k=600, reranker=reranker, reranker_k=200, reranker_weight=0.7, mode="hybrid")
    print("==", qid, query)
    for rank, item in enumerate(ranked[:5], start=1):
        doc = Path(item.get("source") or "").name
        page = item.get("page")
        print(rank, doc, page, round(item.get("score", 0), 3))
    print()
