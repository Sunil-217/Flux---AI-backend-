from openai import OpenAI

from app.core.config import (
    NVIDIA_API_KEY
)

from app.services.chroma_service import (
    get_or_create_collection
)

from app.services.embedding_service import (
    embedding_model
)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)


def ask_question(
    chat_id: str,
    question: str
):

    # Get vector collection
    collection = get_or_create_collection(
        chat_id
    )

    # Embed question
    question_embedding = (
        embedding_model.encode(question)
        .tolist()
    )

    # Retrieve similar chunks
    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=3
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    context = "\n\n".join(
        documents
    )

    # Prompt
    prompt = f"""
You are an AI assistant specialized in
answering questions from uploaded files.

Rules:
- Answer ONLY from provided context
- Do NOT hallucinate
- Be concise and accurate
- If answer is unavailable, clearly say so

Context:
{context}

Question:
{question}
"""

    # NVIDIA response
    completion = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,
        max_tokens=500
    )

    answer = (
        completion
        .choices[0]
        .message
        .content
    )

    return {
        "answer": answer,

        "sources": [
            {
                "content": documents[i],
                "metadata": metadatas[i]
            }
            for i in range(len(documents))
        ]
    }

