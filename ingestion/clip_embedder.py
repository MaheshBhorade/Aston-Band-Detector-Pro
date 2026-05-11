# File: content_detection_system/ingestion/clip_embedder.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import clip
import numpy as np
import logging
from PIL import Image
import faiss

logger = logging.getLogger("ad_ingestion.clip_embedder")


class CLIPEmbedder:

    def __init__(self, model_name="ViT-B/32", batch_size=64):

        self.batch_size = batch_size

        # Detect device automatically
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.warning("CUDA not available, using CPU")

        logger.info(f"Loading CLIP model ({model_name}) on {self.device}")

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        torch.backends.cudnn.benchmark = True
        self.model.eval()

        # Determine embedding dimension
        sample = self.preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(self.device)

        with torch.no_grad():
            vec = self.model.encode_image(sample)

        self.embedding_dim = vec.shape[-1]

        logger.info(f"CLIP embedding dimension: {self.embedding_dim}")

    # ------------------------------------------------
    # Embed Image Paths
    # ------------------------------------------------


    def embed_images(self, frames, is_bgr=True):
        """Embed a list of frames. Set is_bgr=True if frames are BGR (OpenCV default)."""
        images = []

        for frame in frames:
            if is_bgr:
                image = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
            else:
                image = Image.fromarray(frame)  # Already RGB
            image = self.preprocess(image)

            images.append(image)

        embeddings = []

        for i in range(0, len(images), self.batch_size):

            batch = torch.stack(images[i:i+self.batch_size]).to(self.device)

            with torch.no_grad():
                batch_emb = self.model.encode_image(batch)

            batch_emb = batch_emb.cpu().numpy()

            embeddings.append(batch_emb)

        embeddings = np.vstack(embeddings).astype("float32")

        # NOTE: do NOT call faiss.normalize_L2 here.
        # The detector normalises embeddings via PyTorch (embeddings.norm)
        # before the FAISS search. Normalising twice corrupts the dot-product
        # scores when the norms are already close to 1.0.
        return embeddings