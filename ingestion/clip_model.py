import os
os.environ["OMP_NUM_THREADS"] = "1"
import logging
from ingestion.clip_embedder import CLIPEmbedder
# from config.logger_setup import setup_logging

# setup_logging()
logger = logging.getLogger("ad_ingestion.clip_model")

_clip_model = None


def get_clip_model():

    global _clip_model

    if _clip_model is None:
        logger.info("Initializing shared CLIP model")
        _clip_model = CLIPEmbedder()

    return _clip_model