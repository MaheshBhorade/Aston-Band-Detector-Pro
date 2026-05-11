import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import os
import shutil
import logging
from pathlib import Path
import traceback
# from ad_ingestion_worker.ingestion.hash_utils import check_duplicate
# from ad_ingestion_worker.ingestion.hash_utils import find_duplicate
import cv2
from ingestion.hash_utils import compute_video_hash, check_duplicate
from ingestion.frame_extractor import FrameExtractor
from ingestion.frame_filter import FrameFilter
from ingestion.clip_embedder import CLIPEmbedder
from ingestion.prototype_builder import PrototypeBuilder
from ingestion.index_manager import IndexManager
from ingestion.clip_model import get_clip_model
from config.config_loader import load_config
from database.db_manager import DatabaseManager

from dotenv import load_dotenv

load_dotenv(r"D:\ABHI-REX\ABHI-rex 2.0\ad_ingestion\ad_ingestion_worker\.env")

logger = logging.getLogger("ad_ingestion.service")


class AdIngestService:

    def __init__(self):

        # -----------------------------
        # Load configuration
        # -----------------------------
        self.config = load_config()
        self.save_prototypes = self.config["ingestion"]["save_prototype_images"]

        # -----------------------------
        # Paths
        # -----------------------------
        self.incoming_dir = Path("storage/incoming_ads")
        self.processing_dir = Path("storage/processing_ads")
        self.processed_dir = Path("storage/processed_ads")
        self.failed_dir = Path("storage/failed_ads")
        self.duplicate_dir = Path("storage/duplicate_ads")
        # Optional debug prototype folder
        self.prototype_dir = Path("storage/prototypes")

        for d in [
            self.processing_dir,
            self.processed_dir,
            self.failed_dir,
            self.duplicate_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)

        if self.save_prototypes:
            self.prototype_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Modules
        # -----------------------------
        self.frame_extractor = FrameExtractor()
        self.frame_filter = FrameFilter()
        self.embedder = CLIPEmbedder()
        self.prototype_builder = PrototypeBuilder()
        self.index_manager = IndexManager()
        db_url = os.getenv("DATABASE_URL")
        # self.db = DatabaseManager("database/content_detection.db")
        self.db = DatabaseManager(db_url)


    def move_to_duplicate(self, video_path, processing_video, metadata_path):

        duplicate_video = self.duplicate_dir / video_path.name
        duplicate_json = self.duplicate_dir / metadata_path.name

        try:

            # Move video
            if processing_video and Path(processing_video).exists():
                shutil.move(str(processing_video), duplicate_video)

            elif video_path.exists():
                shutil.move(str(video_path), duplicate_video)

            # Move metadata
            if metadata_path.exists():
                shutil.move(str(metadata_path), duplicate_json)

        except Exception as e:
            logger.error(f"Failed moving duplicate files: {e}")    

    # ------------------------------------------------
    # Process One Advertisement
    # ------------------------------------------------

    def process_ad(self, video_path, metadata_path):


        video_path = Path(video_path)
        metadata_path = Path(metadata_path)

        processing_video = None
        ad_id = None
        frames = []

        try:

            logger.info(f"Starting ingestion for {video_path}")

            # -----------------------------
            # Load metadata safely
            # -----------------------------
            with open(metadata_path, "r", encoding="utf-8-sig") as f:
                metadata = json.load(f)

            ad_id = metadata["ad_id"]

            # -----------------------------
            # Move video to processing
            # -----------------------------
            processing_video = self.processing_dir / video_path.name
            shutil.move(str(video_path), processing_video)

            # -----------------------------
            # Compute video hash
            # -----------------------------
            video_hash = compute_video_hash(processing_video)
            duration = self.frame_extractor.get_video_duration(processing_video)
            logger.info(f"VID DURATION = {duration}")
            # -----------------------------
            # Duplicate detection
            # -----------------------------
            # dup = check_duplicate(self.db, ad_id, video_hash)

            # if dup:
            #     dup_type, original_ad_id = dup

            #     logger.warning(
            #         f"Duplicate detected ({dup_type}). original={original_ad_id}, incoming={ad_id}"
            #     )

            #     # Insert the duplicate record first
            #     self.db.insert_advertisement(
            #         metadata,
            #         video_hash,
            #         str(processing_video)
            #     )

            #     # Mark duplicate relationship
            #     self.db.mark_duplicate(ad_id, original_ad_id)

            #     # Move files to processed
            #     processed_video = self.processed_dir / video_path.name
            #     processed_json = self.processed_dir / metadata_path.name

            #     shutil.move(str(processing_video), processed_video)

            #     if metadata_path.exists():
            #         shutil.move(str(metadata_path), processed_json)

            #     logger.info(
            #         f"Duplicate advertisement recorded: {ad_id} → original {original_ad_id}"
            #     )

                # return

            # -----------------------------
            # Insert new advertisement
            # -----------------------------
            self.db.insert_advertisement(
                metadata,
                video_hash,
                str(processing_video)
            )

            # -----------------------------
            # Frame extraction
            # -----------------------------
            frames = self.frame_extractor.extract_frames(
                str(processing_video),
                ad_id
            )

            # -----------------------------
            # Frame filtering
            # -----------------------------
            frames = self.frame_filter.filter_frames(frames)

            if len(frames) == 0:
                raise RuntimeError("No valid frames after filtering")

            # -----------------------------
            # CLIP embeddings
            # -----------------------------
            embeddings = self.embedder.embed_images(frames)

            # -----------------------------
            # Prototype generation
            # -----------------------------
            prototypes, proto_frames = self.prototype_builder.build_prototypes(
                embeddings,
                frames
            )

            # -----------------------------
            # CLIP duplicate detection
            # -----------------------------
            # if self.index_manager.check_duplicate_IM(prototypes):

            #     logger.warning("Duplicate advertisement detected by CLIP similarity")

            #     self.move_to_duplicate(video_path, processing_video, metadata_path)

            #     self.db.update_ad_status(ad_id, "DUPLICATE")

            #     return
            result = self.index_manager.classify_ad(prototypes, duration, ad_id)

            logger.info(f"Classification result: {result}")

            is_variant = False

            # -----------------------------
            # Handle classification result
            # -----------------------------
            if result["type"] == "DUPLICATE":
                logger.warning("Duplicate advertisement detected by CLIP similarity")

                # self.move_to_duplicate(video_path, processing_video, metadata_path)

                # self.db.update_ad_status(ad_id, "DUPLICATE")
                original_ad_id = result["matched_ad_id"]

                logger.warning(
                    f"CLIP duplicate detected → original={original_ad_id}, incoming={ad_id}"
                )

                self.db.mark_duplicate(ad_id, original_ad_id)

                # Move files to processed
                processed_video = self.processed_dir / video_path.name
                processed_json = self.processed_dir / metadata_path.name

                shutil.move(str(processing_video), processed_video)

                if metadata_path.exists():
                    shutil.move(str(metadata_path), processed_json)

                logger.info(
                    f"Duplicate advertisement recorded: {ad_id} → original {original_ad_id}"
                )

                return


            elif result["type"] == "VARIANT":

                original_ad_id = result["matched_ad_id"]

                logger.info(
                    f"Variant detected → base={original_ad_id}, variant={ad_id}"
                )

                is_variant = True
                self.db.update_variant_parent(ad_id, original_ad_id)

            # -----------------------------
            # Update variant flag
            # -----------------------------
            logger.info(f"is_variant={is_variant} for ad {ad_id}")
            self.db.update_is_variant(ad_id, is_variant)
            # -----------------------------
            # Update FAISS index
            # -----------------------------
            vector_ids = self.index_manager.update_index(prototypes)
            frame_paths = [None] * len(vector_ids)
            self.db.insert_prototypes_batch(ad_id,vector_ids,frame_paths)

            # -----------------------------
            # Save prototype mapping
            # -----------------------------
            # for vid, frame in zip(vector_ids, proto_frames):

            #     self.db.insert_prototypes_batch(
            #         ad_id,
            #         vid,
            #         frame_paths
            #     )

            # -----------------------------
            # Optional prototype saving
            # -----------------------------
            if self.save_prototypes:

                proto_dir = self.prototype_dir / ad_id
                proto_dir.mkdir(parents=True, exist_ok=True)

                for i, frame in enumerate(proto_frames):

                    dst = proto_dir / f"proto_{i}.jpg"
                    cv2.imwrite(str(dst),frame)
                    # shutil.copy(frame, dst)

            # -----------------------------
            # Update DB status
            # -----------------------------
            self.db.mark_prototype_built(ad_id)
            self.db.update_ad_status(ad_id, "COMPLETED")

            # -----------------------------
            # Move video + json to processed
            # -----------------------------
            processed_video = self.processed_dir / video_path.name
            processed_json = self.processed_dir / metadata_path.name

            shutil.move(str(processing_video), processed_video)

            if metadata_path.exists():
                shutil.move(str(metadata_path), processed_json)

            logger.info(f"Ingestion completed for {ad_id}")

        except Exception as e:

            logger.error(f"Ingestion failed: {e}")
            logger.error(traceback.format_exc())

            try:

                failed_video = self.failed_dir / video_path.name
                failed_json = self.failed_dir / metadata_path.name

                if processing_video and processing_video.exists():
                    shutil.move(str(processing_video), failed_video)

                if metadata_path.exists():
                    shutil.move(str(metadata_path), failed_json)

            except Exception as move_err:

                logger.error(f"Failed to move files to failed folder: {move_err}")

            if ad_id:
                self.db.update_ad_status(ad_id, "FAILED")

        finally:

            # -----------------------------
            # Clean temporary frame files
            # -----------------------------
            try:

                for f in frames:
                    p = Path(f)

                    if p.exists():
                        p.unlink()

            except Exception:
                pass

