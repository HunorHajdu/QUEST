import faiss
from embedder.embedder import Embedder

class VectorDatabase:
    def __init__(self, vec_dim=512, embedder=None):
        self.index = faiss.IndexFlatL2(vec_dim)
        self.embedder = Embedder() if embedder is None else embedder

    def add_vectors(self, text):
        vectors = self.embedder.embed(text)
        self.index.add(vectors)

    def search_vector(self, text, k=1):
        vector = self.embedder.embed(text)
        return self.index.search(vector, k)