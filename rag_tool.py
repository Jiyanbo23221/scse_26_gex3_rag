from policy_loader import load_policy_documents

def load_documents():
    return load_policy_documents()

def retrieve_relevant_chunks(question, documents, top_k=3):
    question_lower = question.lower()
    keywords = question_lower.split()
    scored = []
    for doc in documents:
        text = doc.page_content.lower()
        topic = doc.metadata.get("topic", "").lower()
        score = 0
        for kw in keywords:
            if kw in text:
                score += 2
            if kw in topic:
                score += 3
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:top_k]]
