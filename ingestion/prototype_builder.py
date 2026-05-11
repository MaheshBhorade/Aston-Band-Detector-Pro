import os
import numpy as np
import logging
import cv2

logger = logging.getLogger("ad_ingestion.prototype_builder")

class PrototypeBuilder:
    def __init__(
        self,
        duplicate_threshold=0.99,
        prototype_similarity_threshold=0.98,
        max_prototypes=10,
        min_prototypes=10
    ):
        self.duplicate_threshold = duplicate_threshold
        self.prototype_similarity_threshold = prototype_similarity_threshold
        self.max_prototypes = max_prototypes
        self.min_prototypes = min_prototypes

    def _normalize_embeddings(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return embeddings / norms

    def get_clarity_score(self, image):
        """Calculates the sharpness of an image using Laplacian variance. Accepts BGR or RGB."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def build_prototypes(self, embeddings, images):
        """
        Picks EXACTLY 10 clearest and most diverse frames.
        """
        if len(images) == 0:
            return [], []

        # 1. Filter by Clarity (Keep the top 50% sharpest frames if we have enough)
        if len(images) > self.max_prototypes * 2:
            clarity_scores = [self.get_clarity_score(img) for img in images]
            threshold = np.percentile(clarity_scores, 50)
            valid_indices = [i for i, score in enumerate(clarity_scores) if score >= threshold]
            
            embeddings = embeddings[valid_indices]
            images = [images[i] for i in valid_indices]

        embeddings = self._normalize_embeddings(embeddings)
        
        # 2. Pick the ACTUAL sharpest frame as the starting point
        clarity_scores = [self.get_clarity_score(img) for img in images]
        best_start = int(np.argmax(clarity_scores))
        selected_indices = [best_start]
        
        # 3. Fill up to exactly max_prototypes
        target_count = min(len(images), self.max_prototypes)
        
        while len(selected_indices) < target_count:
            current_selected = embeddings[selected_indices]
            similarities = np.dot(embeddings, current_selected.T)
            max_sims = np.max(similarities, axis=1)
            
            # Find the most different remaining frame
            next_idx = np.argmin(max_sims)
            
            # Avoid picking the same frame twice
            if next_idx in selected_indices:
                # If we are stuck, just pick the next available index
                for i in range(len(images)):
                    if i not in selected_indices:
                        next_idx = i
                        break
            
            selected_indices.append(next_idx)

        logger.info(f"Generated {len(selected_indices)} high-clarity prototypes.")
        
        proto_embeddings = embeddings[selected_indices]
        proto_frames = [images[i] for i in selected_indices]
        
        return proto_embeddings.astype("float32"), proto_frames

    def _run_kmeans(self, embeddings, images, k=10):
        return self.build_prototypes(embeddings, images)