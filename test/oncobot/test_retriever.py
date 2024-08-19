from src.oncobot.retriever import CustomRetriever


def test_retriever():
    retriever = CustomRetriever(num_docs=10, semantic_ratio=0.1)
    query = "Fake Patient1"
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print(doc.title)
        print("\n")
