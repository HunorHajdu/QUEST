from vector_database.embedder.embedder import Embedder

import numpy as np
import hnswlib

class VectorDatabase:
    def __init__(self, embedder=None, vec_dim=384, documents=None):
        self.embedder = Embedder(vec_dim=vec_dim) if embedder is None else embedder
        self.index = hnswlib.Index(space='cosine', dim=vec_dim)
        self.index.init_index(max_elements=100000, ef_construction=32, M=16)
        self.index.set_ef(16)
        self.documents = documents

    def add_vectors(self, text):
        vectors = self.embedder.embed(text)
        self.index.add_items(vectors, np.arange(len(vectors)))


    def search_vector(self, text, k=1):
        vector = self.embedder.embed(text)
        
        results = []
        labels, distances = self.index.knn_query(vector, k=k)
        for label, distance in zip(labels[0], distances[0]):
            results.append({"label": label, "distance": distance, 'vector': self.index.get_items([label])[0], 'text': self.documents[label]})        

        return results  