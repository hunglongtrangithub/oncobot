from pydantic import BaseModel
from typing import List, Optional

import meilisearch
from meilisearch.errors import MeilisearchApiError

from src.utils.logger_config import logger
from .scripts.index import EMBEDDER_NAME, INDEX_NAME, SEARCH_API_KEY, MEILI_API_URL


class Document(BaseModel):
    id: str
    page_content: str
    title: str
    source: str


def longest_common_substring(s1, s2):
    n = len(s1)
    m = len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    max_length = 0
    ending_index_s1 = -1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    ending_index_s1 = i - 1
            else:
                dp[i][j] = 0

    # Extract the longest common substring using the ending index and max_length
    if max_length <= 0:
        logger.info("No common substring found")
        return "", 0
    longest_common_substring = s1[
        ending_index_s1 - max_length + 1 : ending_index_s1 + 1
    ]

    return longest_common_substring, max_length


class CustomRetriever:
    def __init__(self, num_docs: int = 5, semantic_ratio: float = 0.1):
        self.index_name = INDEX_NAME
        self.embedder_name = EMBEDDER_NAME
        self.num_docs = num_docs
        self.semantic_ratio = semantic_ratio
        self.client = meilisearch.Client(
            url=MEILI_API_URL,
            api_key=SEARCH_API_KEY,
        )
        self.index = self.client.index(self.index_name)

    def get_relevant_documents(self, query: str) -> List[Document]:
        opt_params = {
            "hybrid": {
                "semanticRatio": self.semantic_ratio,
                "embedder": self.embedder_name,
            },
            "limit": self.num_docs,
        }
        query = "".join(query.strip().lower().split())
        logger.info(f"Searching documents for query: {query}")
        try:
            result = self.index.search(query, opt_params)
        except MeilisearchApiError as e:
            logger.error(f"Error searching documents: {e}")
            return []

        docs = result["hits"]
        docs = sorted(
            docs,
            key=lambda doc: longest_common_substring(
                query,
                "".join(
                    doc["title"].lower().split("_")
                ),  # remove underscores. May not be necessary
                # doc["title"].lower(),
            )[1],
            reverse=True,
        )
        logger.info(f"Found {len(docs)} documents: {[doc['title'] for doc in docs]}")
        return [
            Document(
                id=doc["id"],
                page_content=doc["page_content"],
                title=doc["title"],
                source=doc["source"],
            )
            for doc in docs
        ]

    async def aget_relevant_documents(self, query: str):
        return self.get_relevant_documents(query)

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        try:
            doc = self.index.get_document(doc_id)
            return Document(
                id=doc.id,
                page_content=doc.page_content,
                title=doc.title,
                source=doc.source,
            )
        except MeilisearchApiError as e:
            logger.error(f"Error retrieving document with id {doc_id}: {e}.")
            return None
