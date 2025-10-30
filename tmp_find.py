from app.chatbot.modules.embeddings import load_faiss_index
index, records, embeddings = load_faiss_index("app/chatbot/data/faiss_index/index")
for rec in records:
    if rec.get("chunk_index") == 304:
        print(rec.get("source"), rec.get("page"))
        break
else:
    print("not found")
