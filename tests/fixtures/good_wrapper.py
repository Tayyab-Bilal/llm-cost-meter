from langchain_core.embeddings import Embeddings


class MeteredEmbeddings(Embeddings):
    """Subclasses the real base, so isinstance(x, Embeddings) passes."""

    def __init__(self, inner):
        self._inner = inner

    def embed_documents(self, texts):
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        return self._inner.embed_query(text)
