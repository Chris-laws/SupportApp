import numpy as np
from app.chatbot.modules.embeddings import load_faiss_index, get_embeddings
from app.chatbot.modules.retriever import HybridRetriever, rewrite_query_with_llama3
from app.chatbot.modules.reranker import CrossEncoderReranker

index, records, embeddings = load_faiss_index("app/chatbot/data/faiss_index/index")
retriever = HybridRetriever(records, embeddings=embeddings, faiss_index=index)
reranker = CrossEncoderReranker()

query = "Softphone Kennwort aendern"
vec = get_embeddings([query])[0]
ranked = retriever.retrieve(query, np.array(vec, dtype=np.float32), top_k=400, bm25_k=600, reranker=reranker, reranker_k=200, reranker_weight=0.7)
for rank, item in enumerate(ranked, start=1):
    if int(item.get("chunk_index", -1)) == 304:
        print("found at rank", rank)
        break
else:
    print("not found")
