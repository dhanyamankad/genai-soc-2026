# task2.py
"""
Task 2: Cosine Similarity Between Sentences

Embeds two user-provided sentences with sentence-transformers/all-MiniLM-L6-v2
and prints their cosine similarity (0 = unrelated, 1 = identical meaning).
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    model = SentenceTransformer(MODEL_NAME)

    sentence_a = input("Sentence A: ").strip()
    sentence_b = input("Sentence B: ").strip()

    embeddings = model.encode([sentence_a, sentence_b])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    print(f"Cosine Similarity: {similarity:.2f}")


if __name__ == "__main__":
    main()