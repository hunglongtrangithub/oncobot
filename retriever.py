from pydantic import BaseModel
import meilisearch
from meilisearch.errors import MeilisearchApiError
from typing import List
from logger_config import get_logger
from _index import EMBEDDER_NAME, INDEX_NAME, SEARCH_API_KEY, MEILI_API_URL

logger = get_logger(__name__)


class Document(BaseModel):
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
    # print(s1)
    # print(s2)
    if max_length <= 0:
        print("No common substring found")
        return "", 0
    longest_common_substring = s1[
        ending_index_s1 - max_length + 1 : ending_index_s1 + 1
    ]
    # print(f"Longest common substring: {longest_common_substring}")
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
                "".join(doc["title"].lower().split("_")),
                # doc["title"].lower(),
            )[1],
            reverse=True,
        )
        logger.info(f"Found {len(docs)} documents: {[doc['title'] for doc in docs]}")
        return [
            Document(
                page_content=doc["page_content"],
                title=doc["title"],
                source=doc["source"],
            )
            for doc in docs
        ]

    async def aget_relevant_documents(self, query: str):
        return self.get_relevant_documents(query)


if __name__ == "__main__":
    retriever = CustomRetriever(num_docs=5)
    # query = "What is fake patient 1 diagnosed with?"
    query = "fake patient 1"
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print(doc.title)