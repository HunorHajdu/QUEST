from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", vec_dim=384):
        self.model = SentenceTransformer(model_name)
        self.vec_dim = vec_dim
        self.pca = PCA(n_components=vec_dim)

    def embed(self, text):
        vectors = self.model.encode(text, convert_to_numpy=True)
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.vec_dim:
            vectors = self.pca.fit_transform(vectors)

        return vectors