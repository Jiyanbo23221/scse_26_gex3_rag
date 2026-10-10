from ollama import chat
from rag_tool import load_documents, retrieve_relevant_chunks

def answer_question(question):
    documents = load_documents()
    relevant = retrieve_relevant_chunks(question, documents, top_k=3)

    context_parts = []
    for doc in relevant:
        context_parts.append(f"[{doc.metadata.get('policy_id')} - {doc.metadata.get('topic')}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are a university IT support assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say exactly: "I don't know the phone number for the University IT Service Desk."

Context:
{context}

Question: {question}
"""

    response = chat(
        model="qwen3:0.6b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content
