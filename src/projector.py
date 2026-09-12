import numpy as np
from sklearn.decomposition import PCA

class ManifoldProjector:
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(embeddings)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_
