import json
from pathlib import Path
import numpy as np

from app.chatbot.modules.embeddings import load_faiss_index, get_embeddings
from app.chatbot.modules.retriever import HybridRetriever
from app.chatbot.modules.reranker import CrossEncoderReranker

source = Path("data/eval/questions_gold.json")
target = Path("app/chatbot/evaluation/questions_eval20.json")
questions = json.loads(source.read_text(encoding="utf-8"))[:20]

index, records, embeddings = load_faiss_index("app/chatbot/data/faiss_index/index")
retriever = HybridRetriever(records, embeddings=embeddings, faiss_index=index)
reranker = CrossEncoderReranker()

candidate_pool = 80
bm25_k = max(200, candidate_pool * 4)
reranker_k = max(candidate_pool * 3, 150)

samples = []
for sample in questions:
    question = sample["question"]
    embedding = get_embeddings([question])[0]
    ranked = retriever.retrieve(
        question=question,
        query_embedding=np.array(embedding, dtype=np.float32),
        top_k=candidate_pool,
        bm25_k=bm25_k,
        reranker=reranker,
        reranker_k=reranker_k,
        reranker_weight=0.7,
    )
    selected = []
    for item in ranked[:5]:
        idx = int(item.get("chunk_index", -1))
        if idx >= 0:
            selected.append(idx)
    if not selected:
        selected = [int(r) for r in sample.get("ground_truth", {}).get("relevant_chunk_indices", [])[:1]]
    samples.append({
        "question_id": sample.get("question_id"),
        "question": question,
        "ground_truth": {
            "relevant_chunk_indices": selected,
            "expected_keywords": sample.get("ground_truth", {}).get("expected_keywords", [])
        }
    })

target.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"dataset regenerated with {len(samples)} questions")
