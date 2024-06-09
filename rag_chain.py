from jinja2 import Template
from typing import List, Optional, Dict, Tuple, Union, Generator, AsyncGenerator
from pydantic import BaseModel, Field
from transformers import pipeline

from logger_config import get_logger
from retriever import CustomRetriever, Document
from custom_chat_model import BaseChat
import logging

logger = get_logger(__name__)


# System prompt for medical chatbot
SYSTEM_TEMPLATE = """\
You are a medical assistant, tasked with answering any question about the provided \
medical context.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results. You must \
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

If there is no relevant information within the context, just say "Hmm, I'm not sure." \
Do not make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user. Always answer in 80 words or less.\
"""

# System prompt that includes NER results
SYSTEM_TEMPLATE = """\
You are a medical assistant tasked with answering any question about the provided \
context, which includes a medical document and its NER information.

Generate a comprehensive and informative answer of 80 words or less for the given \
question based solely on the provided context. Use an unbiased and journalistic tone. \
Do not repeat text.

If there is no relevant information within the context, just say "Hmm, I'm not sure." \
Do not make up an answer.

Anything between the following `context` HTML blocks is retrieved from a knowledge \
bank, not part of the conversation with the user.

<context>
    {context}
</context>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Always answer in 80 words or less.\
"""


# Template for asking the model to extract the name of the patient
REPHRASE_TEMPLATE = """
Given the following chat history:

{chat_history}

And the follow-up question:

{question}

Extract the name of the patient that is the most relevant to the conversation leading up to the follow-up question. Do not \
include any other information in your response.
"""

# Jinja2 template for formatting chat history
CHAT_TEMPLATE_STRING = """\
{%- for message in messages %}
{% if message.role == 'user' -%}
Human: {{ message.content -}}\n
{%- elif message.role == 'assistant' -%}
AI: {{ message.content -}}\n
{%- else -%}
Unknown Role: {{ message.content -}}\n
{%- endif -%}
{%- endfor -%}\
"""


class ChatRequest(BaseModel):
    question: str = Field(..., examples=["What's the weather like today?"])
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


class RAGChain:
    def __init__(self, retriever: CustomRetriever, chat_llm: BaseChat):
        self.chat_llm = chat_llm
        self.retriever = retriever
        self.ner = pipeline("token-classification", model="d4data/biomedical-ner-all")
        self.chat_logger = logging.getLogger(__name__)
        self.chat_logger.setLevel(logging.INFO)
        handler = logging.FileHandler("chat_history.log")
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.chat_logger.addHandler(handler)

    def format_chat_history(
        self, chat_history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
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

        return formatted_chat_history

    def get_serialized_entities(self, text: str) -> str:
        try:
            entities = self.ner(text, grouped_entities=True)
        except Exception as e:
            logger.error(f"Error in NER Engine: {e}")
            entities = []

        if len(entities) == 0:  # type: ignore
            entity_list = [(text, None, 0)]
        else:
            list_format = []
            index = 0
            entities = sorted(entities, key=lambda x: x["start"])  # type: ignore
            for entity in entities:
                list_format.append((text[index : entity["start"]], None, index))
                entity_category = entity.get("entity") or entity.get("entity_group")
                list_format.append(
                    (
                        text[entity["start"] : entity["end"]],
                        entity_category,
                        entity["start"],
                    )
                )
                index = entity["end"]
            list_format.append((text[index:], None, index))
            entity_list = list_format

        output = []
        running_text, running_category, running_start = None, None, None
        # NOTE: This is a hack to avoid the adjacent_separator
        self.adjacent_separator = ""
        for text, category, start in entity_list:
            if running_text is None:
                running_text = text
                running_category = category
                running_start = start
            elif category == running_category:
                running_text += self.adjacent_separator + text
            elif not text:
                # Skip fully empty item, these get added in processing
                # of dictionaries.
                pass
            else:
                output.append((running_text, running_category, running_start))
                running_text = text
                running_category = category
                running_start = start
        if running_text is not None:
            output.append((running_text, running_category, running_start))

        # group the entities by category
        entity_dict = {}
        for text, category, start in output:
            if not category:
                continue
            category = category.replace("_", " ")
            if category in entity_dict:
                entity_dict[category].append(str((text, start)))
            else:
                entity_dict[category] = [str((text, start))]

        # serialize the entities
        serialized_entities = "NER Entities:\n"
        for category, entities in entity_dict.items():
            serialized_entities += f"{category}: {', '.join(entities)}\n"
        return serialized_entities

    async def aretrieve_documents(self, request: ChatRequest) -> List[Document]:
        chat_history = self.format_chat_history(request.chat_history)
        query_question = request.question

        try:
            template = Template(CHAT_TEMPLATE_STRING)
            serialized_chat_history = template.render(messages=chat_history).strip()
        except Exception as e:
            logger.error("Error occurred while applying template to chat history.")
            serialized_chat_history = ""

        rephrase_question_prompt = REPHRASE_TEMPLATE.format(
            chat_history=serialized_chat_history, question=query_question
        )
        try:
            rephrased_question = await self.chat_llm.ainvoke(
                [{"role": "user", "content": rephrase_question_prompt}],
            )
            query_question = rephrased_question.strip()
            self.chat_logger.info(f"Rephrased question: {query_question}")
        except Exception as e:
            logger.error(
                "Error occurred while invoking chat model to rephrase question."
            )
        try:
            docs = await self.retriever.aget_relevant_documents(query_question)
            self.chat_logger.info(f"Retrieved {len(docs)} documents.")
        except Exception as e:
            logger.error("Error occurred while retrieving documents:", e)
            docs = []
        return docs

    def retrieve_documents(self, request: ChatRequest) -> List[Document]:
        chat_history = self.format_chat_history(request.chat_history)
        query_question = request.question

        try:
            template = Template(CHAT_TEMPLATE_STRING)
            serialized_chat_history = template.render(messages=chat_history).strip()
        except Exception as e:
            logger.error("Error occurred while applying template to chat history")
            serialized_chat_history = ""

        rephrase_question_prompt = REPHRASE_TEMPLATE.format(
            chat_history=serialized_chat_history, question=query_question
        )
        try:
            rephrased_question = self.chat_llm.invoke(
                [{"role": "user", "content": rephrase_question_prompt}],
            )
            query_question = rephrased_question.strip()
            self.chat_logger.info(f"Rephrased question: {query_question}")
        except Exception as e:
            logger.error(
                "Error occurred while invoking chat model to rephrase question",
            )
        try:
            docs = self.retriever.get_relevant_documents(query_question)
            self.chat_logger.info(f"Retrieved {len(docs)} documents.")
        except Exception as e:
            logger.error("Error occurred while retrieving documents:", e)
            docs = []
        return docs

    def get_response_streamer_with_docs(
        self, request: ChatRequest, docs: List[Document]
    ) -> Generator[str, None, None]:
        docs = docs[:1]  # Only use the first document for now
        serialized_docs = "\n".join(
            [
                f"<doc id='{i}'>"
                + doc.page_content
                + "\n"
                + self.get_serialized_entities(doc.page_content)
                + "</doc>"
                for i, doc in enumerate(docs)
            ]
        )

        system_prompt = SYSTEM_TEMPLATE.format(context=serialized_docs)
        formatted_chat_history = self.format_chat_history(request.chat_history)
        current_conversation = [
            {"role": "system", "content": system_prompt},
            *formatted_chat_history,
            {"role": "user", "content": request.question},
        ]
        self.chat_logger.info(f"System prompt:\n{system_prompt}")
        try:
            text_streamer = self.chat_llm.stream(current_conversation)
            return text_streamer  # type: ignore
        except Exception as e:
            logger.error("Error occurred while streaming response", e)
            yield ""

    async def aget_response_streamer_with_docs(
        self, request: ChatRequest, docs: List[Document]
    ) -> AsyncGenerator[str, None]:
        docs = docs[:1]  # Only use the first document for now
        serialized_docs = "\n".join(
            [
                f"<doc id='{i}'>"
                + doc.page_content
                + "\n\n"
                + self.get_serialized_entities(doc.page_content)
                + "</doc>"
                for i, doc in enumerate(docs)
            ]
        )

        system_prompt = SYSTEM_TEMPLATE.format(context=serialized_docs)
        formatted_chat_history = self.format_chat_history(request.chat_history)
        current_conversation = [
            {"role": "system", "content": system_prompt},
            *formatted_chat_history,
            {"role": "user", "content": request.question},
        ]
        self.chat_logger.info(f"System prompt:\n{system_prompt}")
        try:
            text_streamer = await self.chat_llm.astream(current_conversation)
            return text_streamer
        except Exception as e:
            logger.error("Error occurred while streaming response:", e)

            async def empty_streamer():
                yield ""

            return empty_streamer()

    async def astream_log(
        self, request: ChatRequest
    ) -> AsyncGenerator[Tuple[str, str, Union[Dict, str, List]], None]:
        self.chat_logger.info(f"Received request: {request.model_dump_json()}")
        yield "replace", "", {}

        docs = await self.aretrieve_documents(request)
        formatted_docs = [doc.model_dump_json() for doc in docs]
        yield "add", "/logs", {}
        yield "add", "/logs/FindDocs", {}
        yield "add", "/logs/FindDocs/final_output", {"output": formatted_docs}

        text_streamer = await self.aget_response_streamer_with_docs(request, docs)
        response = ""
        yield "add", "/streamed_output", []
        async for chunk in text_streamer:
            response += chunk
            yield "add", "/streamed_output/-", chunk
        self.chat_logger.info(f"Generated response: {response}")
        yield "replace", "/final_output", {"output": response}

    def stream_log(
        self, request: ChatRequest
    ) -> Generator[Tuple[str, str, Union[Dict, str, List]], None, None]:
        self.chat_logger.info(f"Received request: {request.model_dump_json()}")
        docs = self.retrieve_documents(request)
        formatted_docs = [doc.model_dump_json() for doc in docs]
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
        self.chat_logger.info(f"Generated response: {response}")
        yield "replace", "/final_output", {"output": response}

    async def ainvoke_log(self, request: ChatRequest) -> Dict[str, Union[str, List]]:
        self.chat_logger.info(f"Received request: {request.model_dump_json()}")
        docs = await self.aretrieve_documents(request)
        text_streamer = await self.aget_response_streamer_with_docs(request, docs)
        response = ""
        async for chunk in text_streamer:
            response += chunk
        self.chat_logger.info(f"Generated response: {response}")
        return {"response": response, "docs": [doc.model_dump_json() for doc in docs]}

    def invoke_log(self, request: ChatRequest) -> Dict[str, Union[str, List]]:
        self.chat_logger.info(f"Received request: {request.model_dump_json()}")
        docs = self.retrieve_documents(request)
        text_streamer = self.get_response_streamer_with_docs(request, docs)
        response = ""
        for chunk in text_streamer:
            response += chunk
        self.chat_logger.info(f"Generated response: {response}")
        return {"response": response, "docs": [doc.model_dump_json() for doc in docs]}


if __name__ == "__main__":
    from retriever import CustomRetriever
    from custom_chat_model import CustomChatHuggingFace

    retriever = CustomRetriever()
    chat_llm = CustomChatHuggingFace()
    rag_chain = RAGChain(retriever, chat_llm)
    # text = open("./docs/fake_patient1/fake_patient1_doc10_NOTE.txt").read()
    # entities = rag_chain.get_serialized_entities(text)
    # print(entities)
    request = ChatRequest(
        question="What is Fake Patient1's diagnosis?",
    )
    response = rag_chain.invoke_log(request)
