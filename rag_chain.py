from pathlib import Path
import asyncio

from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.document import Document
from custom_chat_model import CustomChatModel, CustomChatOpenAI
from custom_chat_model import Message
from jinja2 import Template

from typing import List, Optional, Dict, Tuple, Union, Generator, AsyncGenerator
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., example="What's the weather like today?")
    chat_history: Optional[List[Dict[str, str]]] = None

    class Config:
        schema_extra = {
            "example": {
                "question": "Tell me more about the solar system.",
                "chat_history": [
                    {"human": "hello", "ai": "Hello! How can I assist you today?"},
                    {
                        "human": "What's the largest planet in our solar system?",
                        "ai": "The largest planet in our solar system is Jupiter.",
                    },
                ],
            }
        }


SYSTEM_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""
REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


CHAT_TEMPLATE_STRING = """{% for message in messages %}{% if message['human'] is defined %}Human: {{ message['human'] }}\n{% endif %}{% if message['ai'] is defined %}AI: {{ message['ai'] }}\n{% endif %}{% endfor %}"""

# CHECKPOINT = "facebook/opt-125m"
# CHECKPOINT = "meta-llama/Llama-2-70b-chat-hf"
# chat_llm = CustomChatModel(CHECKPOINT)

# from llm_llama.model_generator.llm_pipeline import load_fine_tuned_model
# CHECKPOINT = Path(__file__).parent / "llm_llama/Llama-2-7b-chat_peft_128"
# model, tokenizer = load_fine_tuned_model(CHECKPOINT, peft_model=1)
# chat_llm = CustomChatModel(model=model, tokenizer=tokenizer)

chat_llm = CustomChatOpenAI()

NUM_DOCUMENTS = 6
vectorstore = FAISS.load_local(
    str(Path(__file__).parent / "faiss_index"),
    embeddings=OpenAIEmbeddings(chunk_size=200),
)
retriever = vectorstore.as_retriever(search_kwargs=dict(k=NUM_DOCUMENTS))


class RAGChain:
    def __init__(self, retriever, chat_llm):
        self.retriever = retriever
        self.chat_llm = chat_llm

    def format_chat_history(
        self, chat_history: Optional[List[Dict[str, str]]]
    ) -> List[Message]:
        if not chat_history:
            return []
        formatted_chat_history = []
        for message in chat_history:
            if "human" not in message or "ai" not in message:
                raise ValueError(
                    "Each message in the chat history must have a 'human' and 'ai' key"
                )
            human_message = {"role": "user", "content": message["human"]}
            ai_message = {"role": "assistant", "content": message["ai"]}
            formatted_chat_history.append(human_message)
            formatted_chat_history.append(ai_message)

        formatted_chat_history = [
            Message(**message) for message in formatted_chat_history
        ]
        return formatted_chat_history

    async def aretrieve_documents(self, request: ChatRequest) -> List[Document]:
        chat_history = request.chat_history or []
        question = request.question

        if chat_history:
            template = Template(CHAT_TEMPLATE_STRING)
            serialized_chat_history = template.render(messages=chat_history)

            rephrase_question_prompt = REPHRASE_TEMPLATE.format(
                chat_history=serialized_chat_history, question=question
            )
            rephrased_question = self.chat_llm(
                [Message(role="user", content=rephrase_question_prompt)],
                stream=False,
            ).strip()
            question = rephrased_question

        docs = await retriever.aget_relevant_documents(question)
        return docs

    def retrieve_documents(self, request: ChatRequest) -> List[Document]:
        chat_history = request.chat_history or []
        question = request.question

        if chat_history:
            template = Template(CHAT_TEMPLATE_STRING)
            serialized_chat_history = template.render(messages=chat_history)

            rephrase_question_prompt = REPHRASE_TEMPLATE.format(
                chat_history=serialized_chat_history, question=question
            )
            rephrased_question = self.chat_llm(
                [Message(role="user", content=rephrase_question_prompt)],
                stream=False,
            ).strip()
            question = rephrased_question

        docs = retriever.get_relevant_documents(question)
        return docs

    def get_response_streamer_with_docs(
        self, request: ChatRequest, docs: List[Document]
    ) -> Generator[str, None, None]:
        serialized_docs = "\n".join(
            [f"<doc id='{i}'>{doc.page_content}</doc>" for i, doc in enumerate(docs)]
        )

        system_prompt = SYSTEM_TEMPLATE.format(context=serialized_docs)
        formatted_chat_history = self.format_chat_history(request.chat_history)
        current_conversation = [
            Message(role="system", content=system_prompt),
            *formatted_chat_history,
            Message(role="user", content=request.question),
        ]

        text_streamer = self.chat_llm(current_conversation, stream=True)
        return text_streamer

    async def astream_log(
        self, request: ChatRequest
    ) -> AsyncGenerator[Tuple[str, str, Union[Dict, str, List]], None]:
        yield "replace", "", {}

        docs = await self.aretrieve_documents(request)
        formatted_docs = [doc.json() for doc in docs]
        yield "add", "/logs", {}
        yield "add", "/logs/FindDocs", {}
        yield "add", "/logs/FindDocs/final_output", {"output": formatted_docs}

        text_streamer = self.get_response_streamer_with_docs(request, docs)
        response = ""
        yield "add", "/streamed_output", []
        for chunk in text_streamer:
            response += chunk
            yield "add", "/streamed_output/-", chunk

        yield "replace", "/final_output", {"output": response}

    def stream_log(
        self, request: ChatRequest
    ) -> Generator[Tuple[str, str, Union[Dict, str, List]], None, None]:
        docs = self.retrieve_documents(request)
        formatted_docs = [doc.json() for doc in docs]
        text_streamer = self.get_response_streamer_with_docs(request, docs)

        yield "replace", "", {}
        yield "add", "/logs", {}
        yield "add", "/logs/FindDocs", {}
        yield "add", "/logs/FindDocs/final_output", {"output": formatted_docs}

        response = ""
        yield "add", "/streamed_output", []
        for chunk in text_streamer:
            response += chunk
            yield "add", "/streamed_output/-", chunk

        yield "replace", "/final_output", {"output": response}


class RAGWithoutMemoryChain:
    def __init__(self, retriever, chat_llm):
        self.retriever = retriever
        self.chat_llm = chat_llm

    async def aretrieve_documents(self, request: ChatRequest) -> List[Document]:
        question = request.question
        docs = await retriever.aget_relevant_documents(question)
        return docs

    def retrieve_documents(self, request: ChatRequest) -> List[Document]:
        question = request.question
        docs = retriever.get_relevant_documents(question)
        return docs

    def get_response_streamer_with_docs(
        self, request: ChatRequest, docs: List[Document]
    ) -> Generator[str, None, None]:
        serialized_docs = "\n".join([f"<doc id='{i}'>{doc.page_content}</doc>" for i, doc in enumerate(docs)])

        system_prompt = SYSTEM_TEMPLATE.format(context=serialized_docs)
        current_conversation = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=request.question),
        ]

        text_streamer = self.chat_llm(current_conversation, stream=True)
        return text_streamer

    async def _arun(self, func, *args, **kwargs):
        """Run a synchronous function in the background thread pool."""
        loop = asyncio.get_running_loop()
        # None uses the default executor (ThreadPoolExecutor)
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def aget_response_streamer_with_docs(self, request, docs):

        # Execute the synchronous function in a separate thread
        # and wait for it to complete without blocking the event loop.
        sync_generator = await self._arun(
            self.get_response_streamer_with_docs, request, docs
        )

        # Now, adapt the synchronous generator for asynchronous iteration
        async def async_generator_wrapper(sync_gen):
            while True:
                try:
                    # Run the next() operation of the sync generator in a separate thread
                    item = await self._arun(next, sync_gen, StopIteration)
                    # If StopIteration is used as a sentinel value by _arun to indicate completion
                    if item is StopIteration:
                        break
                    yield item
                except StopIteration:
                    # Break the loop if the generator is exhausted
                    break

        return async_generator_wrapper(sync_generator)

    def stream_log(
        self, request: ChatRequest
    ) -> Generator[Tuple[str, str, Union[Dict, str, List]], None, None]:
        docs = self.retrieve_documents(request)
        formatted_docs = [doc.json() for doc in docs]
        text_streamer = self.get_response_streamer_with_docs(request, docs)

        yield "replace", "", {}
        yield "add", "/logs", {}
        yield "add", "/logs/FindDocs", {}
        yield "add", "/logs/FindDocs/final_output", {"output": formatted_docs}

        response = ""
        yield "add", "/streamed_output", []
        for chunk in text_streamer:
            response += chunk
            yield "add", "/streamed_output/-", chunk

        yield "replace", "/final_output", {"output": response}

    async def astream_log(
        self, request: ChatRequest
    ) -> AsyncGenerator[Tuple[str, str, Union[Dict, str, List]], None]:
        yield "replace", "", {}

        # Retrieve documents first
        docs = await self.aretrieve_documents(request)
        formatted_docs = [doc.json() for doc in docs]
        yield "add", "/logs", {}
        yield "add", "/logs/FindDocs", {}
        yield "add", "/logs/FindDocs/final_output", {"output": formatted_docs}

        # Use the asynchronous version to get the response streamer with the retrieved docs
        # Assuming get_response_streamer_with_docs_async is an async generator
        response = ""
        yield "add", "/streamed_output", []
        async for chunk in await self.aget_response_streamer_with_docs(request, docs):
            response += chunk
            yield "add", "/streamed_output/-", chunk

        yield "replace", "/final_output", {"output": response}


chain = RAGChain(retriever, chat_llm)
# chain_without_memory = RAGWithoutMemoryChain(retriever, chat_llm)
