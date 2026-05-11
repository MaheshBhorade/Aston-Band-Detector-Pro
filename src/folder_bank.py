import os
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FolderImageBank:
    def __init__(self, engine, root_dir):
        self.engine = engine
        self.root_dir = Path(root_dir)
        self.labels = []
        self.paths = []
        self.features = None
        self.visual_features = None

    def load(self):
        images = []
        visual_features = []
        labels = []
        paths = []

        if not self.root_dir.exists():
            print(f"Bank folder not found: {self.root_dir}")
            return

        for class_dir in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(rgb))
                visual_features.append(self._get_visual_feature(image))
                labels.append(class_dir.name)
                paths.append(str(image_path))

        if not images:
            print(f"No images found in bank: {self.root_dir}")
            return

        self.features = self._embed_images(images)
        self.visual_features = np.vstack(visual_features).astype("float32")
        self.labels = labels
        self.paths = paths
        print(f"Loaded {len(self.labels)} images from {self.root_dir}")

    def _get_visual_feature(self, image):
        resized = cv2.resize(image, (256, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype("float32")
        gray -= gray.mean()
        norm = np.linalg.norm(gray)
        if norm > 0:
            gray /= norm
        return gray.reshape(1, -1)

    def _embed_images(self, images, batch_size=64):
        batches = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                inputs = torch.stack(
                    [self.engine.preprocess(img) for img in images[i:i + batch_size]]
                ).to(self.engine.device)
                embeddings = self.engine.model.encode_image(inputs)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                batches.append(embeddings.cpu().numpy().astype("float32"))

        features = np.vstack(batches)
        faiss.normalize_L2(features)
        return features

    @property
    def is_ready(self):
        return self.features is not None and len(self.labels) > 0

    def search_features(self, query_features, top_k=2):
        if not self.is_ready:
            return []

        query_features = np.asarray(query_features).astype("float32")
        faiss.normalize_L2(query_features)

        scores = query_features @ self.features.T
        top_k = min(top_k, scores.shape[1])
        results = []
        for row_idx in range(scores.shape[0]):
            ordered = np.argsort(-scores[row_idx])
            row = []
            seen_labels = set()
            for idx in ordered:
                label = self.labels[idx]
                if label in seen_labels:
                    continue

                row.append(
                    {
                        "label": label,
                        "score": float(scores[row_idx, idx]),
                        "path": self.paths[idx],
                    }
                )
                seen_labels.add(label)
                if len(row) >= top_k:
                    break

            results.append(row)

        return results

    def search_visual(self, crops, top_k=2):
        if self.visual_features is None or len(self.labels) == 0:
            return []

        query_features = np.vstack([self._get_visual_feature(crop) for crop in crops]).astype("float32")
        scores = query_features @ self.visual_features.T

        results = []
        for row_idx in range(scores.shape[0]):
            ordered = np.argsort(-scores[row_idx])
            row = []
            seen_labels = set()
            for idx in ordered:
                label = self.labels[idx]
                if label in seen_labels:
                    continue

                row.append(
                    {
                        "label": label,
                        "score": float(scores[row_idx, idx]),
                        "path": self.paths[idx],
                    }
                )
                seen_labels.add(label)
                if len(row) >= top_k:
                    break

            results.append(row)

        return results
