"""
Извлечение эмбеддингов SigLIP + UMAP + KMeans для деления игроков на команды.
"""

import numpy as np
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
import supervision as sv

from .constants import (
    SIGLIP_MODEL_NAME,
    UMAP_N_COMPONENTS,
    KMEANS_N_CLUSTERS,
    DEFAULT_DEVICE,
    DEFAULT_BATCH_SIZE
)

class TeamClassifier:
    """
    Unsupervised team classifier.
    Экстракт эмбеддингов SigLIP -> UMAP (n_components=...) -> KMeans (n_clusters=2).
    """

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        self.device = device
        self.batch_size = batch_size

        # SigLIP from HuggingFace
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_NAME
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)

        # UMAP reducer
        self.reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS)

        # KMeans clustering
        self.cluster_model = KMeans(n_clusters=KMEANS_N_CLUSTERS)

    def extract_features(self, crops: list[np.ndarray]) -> np.ndarray:
        """
        Конвертация списка кропов (OpenCV -> PIL) -> эмбеддинги.
        """
        pil_imgs = [sv.cv2_to_pillow(c) for c in crops]
        batches = [
            pil_imgs[i : i + self.batch_size]
            for i in range(0, len(pil_imgs), self.batch_size)
        ]
        feats = []

        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(images=batch, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                out = self.features_model(**inputs)
                emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
                feats.append(emb)

        return np.vstack(feats) if feats else np.empty((0,))

    def fit(self, crops: list[np.ndarray]) -> None:
        """
        Fit UMAP + KMeans по списку кропов.
        Сохраняет raw_data, чтобы transform работал без ошибо
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        # сохраняем, иначе transform() упадет, ежели fit был на единственном sample
        self.reducer._raw_data = data
        self.cluster_model.fit(projections)

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """
        Предсказание team_id для новых кропов
        """
        if not crops:
            return np.array([], dtype=int)

        data = self.extract_features(crops)
        proj = self.reducer.transform(data)
        return self.cluster_model.predict(proj).astype(int)