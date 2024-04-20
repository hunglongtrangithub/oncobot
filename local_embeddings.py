from typing import List

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_id):
        # Should use the GPU by default
        self.model = SentenceTransformer(model_id)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using a locally running
           Hugging Face Sentence Transformer model

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = self.model.encode(texts)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a locally running HF
        Sentence transformer.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self.model.encode(text)
        return list(map(float, embedding))


if __name__ == "__main__":
    model_id = "sentence-transformers/paraphrase-albert-small-v2"
    embeddings = LocalHuggingFaceEmbeddings(model_id)
    texts = ["This is a test", "This is another test"]
    print(embeddings.embed_documents(texts))
    print(embeddings.embed_query("This is a test"))
