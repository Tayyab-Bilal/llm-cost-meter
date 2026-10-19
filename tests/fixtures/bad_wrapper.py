class MeteredEmbeddings:
    """Looks like an embeddings client but has no base class: the SDK's isinstance check rejects it."""

    def __init__(self, inner):
        self._inner = inner

    def embed_documents(self, texts):
        return self._inner.embed_documents(texts)
