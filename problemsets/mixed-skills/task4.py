# task4.py
"""
Task 4: Simple RAG with an In-Memory Vector Store (no ChromaDB)

Embeds a small set of documents, finds the most similar one to a user query
via cosine similarity, then asks Groq to answer using ONLY that document.
"""
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "openai/gpt-oss-120b"

DOCUMENTS = [
    "Python is a high-level programming language.",
    "LLMs are trained on vast amounts of text data.",
    "RAG combines retrieval and generation to answer questions.",
    "Embeddings map text to dense vectors for semantic search.",
    "Groq provides extremely fast LLM inference.",
]


def build_in_memory_store(documents, embed_model):
    """Returns a list of (text, embedding) tuples — the 'vector store'."""
    embeddings = embed_model.encode(documents)
    return list(zip(documents, embeddings))


def retrieve_most_similar(query, store, embed_model):
    """Embeds the query and returns the single most similar document's text."""
    query_embedding = embed_model.encode([query])[0]
    texts = [text for text, _ in store]
    embeddings = [emb for _, emb in store]

    similarities = cosine_similarity([query_embedding], embeddings)[0]
    best_idx = similarities.argmax()

    return texts[best_idx], similarities[best_idx]


def answer_with_context(query, context, client):
    system_prompt = (
        "Answer the user's question using ONLY the following context. "
        "If the answer is not in the context, say 'I don't know.'\n\n"
        f"Context: {context}"
    )
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    return completion.choices[0].message.content


def main():
    embed_model = SentenceTransformer(EMBED_MODEL)
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    store = build_in_memory_store(DOCUMENTS, embed_model)

    query = input("Query: ").strip()
    best_doc, score = retrieve_most_similar(query, store, embed_model)

    print(f"\nRetrieved (similarity={score:.2f}): {best_doc}")

    answer = answer_with_context(query, best_doc, groq_client)
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()