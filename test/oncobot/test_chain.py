import asyncio
import time
from contextlib import contextmanager
import pytest

from src.oncobot.rag_chain import RAGChain
from src.oncobot.rag_chain import ChatRequest
from src.oncobot.retriever import CustomRetriever
from src.oncobot.custom_chat_model import DummyChat
from src.oncobot.ner import DummyNERProcessor


@contextmanager
def timeit():
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"Time taken: {end - start:.2f} seconds")


@pytest.fixture(scope="module")
def chain():
    default_message = "Fake Patient 3 is diagnosed with stage 2 invasive ductal carcinoma of the right breast, metastatic to right axillary lymph nodes."
    chat_model = DummyChat(default_message=default_message)
    retriever = CustomRetriever(num_docs=5, semantic_ratio=0.1)
    ner = DummyNERProcessor()
    chain = RAGChain(retriever, chat_model, ner)
    return chain


question = "What is Fake Patient1 diagnosed with?"


@pytest.fixture(scope="module", autouse=True)
def messages():
    current_chat = {
        "question": "Can you give me examples of applications built with LangChain?",
        "chat_history": [
            {
                "human": "What is LangChain?",
                "ai": "LangChain is a framework designed to facilitate the building of applications that leverage large language models (LLMs) like GPT. It provides tools and abstractions to make it easier to create complex systems involving language understanding and generation.",
            },
            {
                "human": "How does LangChain work?",
                "ai": "LangChain works by providing a set of libraries and APIs that allow developers to easily integrate language models into their applications. It supports various components for dialogue management, document retrieval, and information synthesis, making it versatile for a range of use cases.",
            },
            {
                "human": "What are some key features of LangChain?",
                "ai": "Key features of LangChain include modular components for building chat systems, document retrieval and summarization tools, and integration capabilities with multiple language models and data sources. It's designed to be flexible and extensible, allowing developers to customize their applications as needed.",
            },
            {
                "human": "Can LangChain be used for building chatbots?",
                "ai": "Yes, LangChain can be used for building chatbots. It offers specialized components for handling conversations, managing context, and generating responses, making it a powerful tool for developers looking to create advanced chatbot applications.",
            },
            {
                "human": "How can LangChain integrate with other technologies?",
                "ai": "LangChain can integrate with other technologies through its flexible architecture. It supports connecting with various APIs, data sources, and language models, allowing for seamless integration with existing systems and tools. This makes it suitable for a wide range of applications beyond just language processing.",
            },
        ],
    }

    messages = ChatRequest(**current_chat)
    return messages


def test_chain(chain):
    with timeit():
        for chunk in chain.stream_log(messages):
            print(chunk, end="\n", flush=True)


def test_chain_async(chain):
    async def print_stream(messages):
        async for chunk in chain.astream_log(messages):
            print(chunk, end="\n", flush=True)

    with timeit():
        asyncio.run(print_stream(messages))


def test_retrieve_docs(chain):
    with timeit():
        docs = chain.retrieve_documents(messages)
    docs = [doc.model_dump_json() for doc in docs]
    import json

    for doc in docs:
        print(json.dumps(doc, indent=4))


def test_retriever(chain):
    with timeit():
        docs = chain.retriever.get_relevant_documents(question)
    docs = [doc.model_dump_json() for doc in docs]
    import json

    for doc in docs:
        print(json.dumps(doc, indent=4))
