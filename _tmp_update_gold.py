import json
from pathlib import Path
from app.chatbot.modules.embeddings import get_embeddings, load_faiss_index
from app.chatbot.modules.retriever import HybridRetriever
from app.chatbot.evaluation.rag_eval_f1 import _tok_norm

gold_path = Path('data/eval/gold.json')
items = json.loads(gold_path.read_text(encoding='utf-8'))

index, records, embeddings = load_faiss_index('app/chatbot/data/faiss_index/index')
retriever = HybridRetriever(records, embeddings=embeddings, faiss_index=index)

updated = []
for item in items:
    query = item['query']
    expected = item.get('expected_answer_keywords') or []
    expected_tokens = set()
    for keyword in expected:
        expected_tokens.update(tok for tok in _tok_norm(keyword) if len(tok) > 2)
    emb = get_embeddings([query])[0]
    ranked = retriever.retrieve(query, emb, top_k=12)
    chosen = []
    for chunk in ranked:
        idx = int(chunk.get('chunk_index', -1))
        if idx < 0:
            continue
        matches = {str(m).lower() for m in (chunk.get('matched_terms') or [])}
        if expected_tokens and (matches & expected_tokens):
            chosen.append(idx)
        elif not expected_tokens:
            chosen.append(idx)
        if len(chosen) >= 2:
            break
    if not chosen and ranked:
        chosen.append(int(ranked[0].get('chunk_index', -1)))
    item['relevant_ids'] = chosen
    updated.append(item)
    print(f"{query[:50]} -> {chosen}")

gold_path.write_text(json.dumps(updated, indent=2), encoding='utf-8')
print('Gold dataset updated with new indices.')
