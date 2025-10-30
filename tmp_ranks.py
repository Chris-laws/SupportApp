import numpy as np
from pathlib import Path
from app.chatbot.modules.embeddings import load_faiss_index, get_embeddings
from app.chatbot.modules.retriever import HybridRetriever
from app.chatbot.modules.reranker import CrossEncoderReranker

index, records, embeddings = load_faiss_index("app/chatbot/data/faiss_index/index")
retriever = HybridRetriever(records, embeddings=embeddings, faiss_index=index)
reranker = CrossEncoderReranker()

queries = {
    "G004": "Softphone vollstaendig beenden",
    "G005": "Softphone Fusion Leiste Funktionen",
    "G006": "Softphone praefiertes Endgeraet wechseln",
    "G008": "Softphone Rufnummernanzeige steuern (CLIR)",
    "G009": "Softphone Rufnummer fuer CLIR deaktivieren",
    "G010": "Softphone Rufnummernanzeige fuer Kontakte anpassen",
    "G013": "HP Drucker Duplexmodul verwenden",
    "G014": "HP Drucker Patronen reinigen",
    "G015": "Abwesenheitsagent in Clerk anlegen"
}
for qid, query in queries.items():
    vec = get_embeddings([query])[0]
    ranked = retriever.retrieve(query, np.array(vec, dtype=np.float32), top_k=150, bm25_k=600, reranker=reranker, reranker_k=200, reranker_weight=0.7, mode="hybrid")
    print("==", qid, query)
    for rank, item in enumerate(ranked[:5], start=1):
        doc = Path(item.get("source") or "").name
        page = item.get("page")
        print(rank, doc, page, round(item.get("score",0),3))
    print()
