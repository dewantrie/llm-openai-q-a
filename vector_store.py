import os
import numpy as np
import faiss

from typing import Callable, Iterable, List, Any, cast

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "vector_store.bin"
DEFAULT_DIMESIONS = 1536 # dimensions of text-ada-embedding-002

class VectorStore:
    """Vector Store.

    Embeddings are stored within a Faiss index.
    Example:
        .. code-block:: python

        from embedding import Embedding
        store = VectorStore(Embedding.embed)
        store.add_texts(['xxx'])
        store.persist()    
    """

    def __init__(self, embedding_function: Callable):
        self._faiss_index = faiss.IndexFlatIP(DEFAULT_DIMESIONS)
        self.embedding_function = embedding_function

    def add_texts(self,
                  texts: Iterable[str],
    ) -> List[str]:
        embeddings = [self.embedding_function(text) for text in texts]
        return self.__add(embeddings)

    def __add(self, embeddings: Iterable[List[float]]) -> List[str]:
        embedding_np = np.array(embeddings, dtype=np.float32)
        for vector in embedding_np:
            self._faiss_index.add(vector)

    def persist(self):
        persist_path: str = os.path.join(DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME)
        dirpath = os.path.dirname(persist_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        faiss.write_index(self._faiss_index, persist_path)
