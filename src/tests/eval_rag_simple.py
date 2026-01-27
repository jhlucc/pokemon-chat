import asyncio

from langchain_openai import ChatOpenAI

from src.core.settings import settings
from src.knowledge.store.vector import VectorStore

# Simple "LLM-as-a-Judge" Eval Script


async def evaluate_rag():
    print("=== Starting RAG Evaluation ===")

    # 1. Setup
    try:
        store = VectorStore(collection_name=settings.database.milvus_collection_name, connection_alias="eval")
    except Exception:
        print("Vector Store not available.")
        return

    eval_llm = ChatOpenAI(
        model=settings.llm.model_name, api_key=settings.llm.api_key, base_url=settings.llm.api_base, temperature=0
    )

    # 2. Test Cases (Question, Expected Key Fact)
    test_cases = [
        ("What does Pikachu evolve into?", "Raichu"),
        ("What is the type of Charmander?", "Fire"),
        ("Where is Pallet Town located?", "Kanto"),
    ]

    score = 0
    total = len(test_cases)

    for q, fact in test_cases:
        print(f"\nQ: {q}")
        # Retrieve
        docs = store.search(q, top_k=3)
        context = "\n".join([d.page_content for d in docs])

        # Grade (Context Precision)
        # Ask LLM if context contains the fact
        prompt = f"""
        Fact: {fact}
        Context:
        {context}

        Does the context contain the fact? Answer YES or NO.
        """
        try:
            grade = eval_llm.invoke(prompt).content.strip().upper()
        except Exception:
            grade = "ERROR"

        print(f"Context Found: {len(docs)} docs")
        print(f"Contains Fact '{fact}'? {grade}")

        if "YES" in grade:
            score += 1

    print("===============================")
    print(f"Final Score: {score}/{total} (Context Precision)")


if __name__ == "__main__":
    import asyncio

    asyncio.run(evaluate_rag())
