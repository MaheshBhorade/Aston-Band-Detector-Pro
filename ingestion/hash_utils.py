# File: content_detection_system/ingestion/hash_utils.py

import hashlib
import os
import logging
import os
os.environ["OMP_NUM_THREADS"] = "1"

logger = logging.getLogger("ad_ingestion.hash_utils")


def compute_video_hash(video_path: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of a video file.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    chunk_size : int
        Size of chunks read from file.

    Returns
    -------
    str
        SHA256 hash string.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    sha256_hash = hashlib.sha256()

    try:
        with open(video_path, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                sha256_hash.update(data)

        hash_value = sha256_hash.hexdigest()

        logger.info(f"Computed video hash for {video_path}: {hash_value}")

        return hash_value

    except Exception as e:
        logger.error(f"Failed to compute hash for {video_path}: {str(e)}")
        raise


def get_file_size(video_path: str) -> float:
    """
    Return video file size in MB (for logging purposes).
    """

    try:
        size_bytes = os.path.getsize(video_path)
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)

    except Exception:
        return 0.0


def check_duplicate(db_manager, ad_id: str, video_hash: str):
    """
    Returns:
        ("VIDEO", existing_ad_id)
        ("AD_ID", existing_ad_id)
        None
    """

    try:

        # check duplicate video
        existing_by_hash = db_manager.get_ad_by_hash(video_hash)
        if existing_by_hash:
            logger.warning(
                f"Duplicate video detected. Existing ad: {existing_by_hash}"
            )
            return ("VIDEO", existing_by_hash)

        # check duplicate ad_id
        if db_manager.ad_id_exists(ad_id):
            logger.warning(
                f"Duplicate ad_id detected: {ad_id}"
            )
            return ("AD_ID", ad_id)

        return None

    except Exception as e:
        logger.error(f"Duplicate check failed: {str(e)}")
        raise

