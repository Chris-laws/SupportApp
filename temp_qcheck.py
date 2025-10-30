from app.chatbot.modules.embeddings import load_faiss_index, get_embeddings
from app.chatbot.modules.retriever import HybridRetriever
index, records, embeddings = load_faiss_index('app/chatbot/data/faiss_index/index')
retriever = HybridRetriever(records, embeddings=embeddings, faiss_index=index)
question = 'Softphone Fusion Leiste Funktionen'
queries = ['Softphone Fusion Leiste Funktionen']
from app.chatbot.modules.retriever import generate_multi_queries
queries = generate_multi_queries(question, total_variants=6)
print('queries:', queries)
for q in queries:
    emb = get_embeddings([q])[0]
    results = retriever.retrieve(question=q, query_embedding=emb, top_k=50, bm25_k=200)
    pages = [(i+1, chunk.get('page'), chunk.get('bm25_score'), chunk.get('dense_score')) for i, chunk in enumerate(results[:20])]
    print('---', q)
    print(pages[:10])
