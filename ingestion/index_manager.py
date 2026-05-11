import math
import os
import threading
os.environ["OMP_NUM_THREADS"] = "1"
import faiss
import numpy as np
import logging
from pathlib import Path
from filelock import FileLock
from database.db_manager import DatabaseManager
from dotenv import load_dotenv
from collections import Counter


load_dotenv(r"D:\ABHI-REX\ABHI-rex 2.0\ad_ingestion\ad_ingestion_worker\.env")

logger = logging.getLogger("ad_ingestion.index_manager")
faiss_lock = threading.Lock()


class IndexManager:

    def __init__(self, index_dir="indexes", dimension=512):

        self.index_dir = Path(index_dir)
        self.dimension = dimension

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.index_dir / "ads.faiss"
        self.index = self.load_index()
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        db_url = os.getenv("DATABASE_URL")
        self.db = DatabaseManager(db_url)
    # ------------------------------------------------
    # Create FAISS index
    # ------------------------------------------------

    def _create_index(self):

        logger.info("Creating new FAISS HNSW index (cosine similarity)")

        # IMPORTANT: metric defined in constructor
        base_index = faiss.IndexHNSWFlat(
            self.dimension,
            32,
            faiss.METRIC_INNER_PRODUCT
        )

        index = faiss.IndexIDMap2(base_index)

        return index

    # ------------------------------------------------
    # Load Index
    # ------------------------------------------------

    def load_index(self):

        try:

            if not self.index_path.exists():

                logger.info("Creating new FAISS index")

                index = self._create_index()

                faiss.write_index(index, str(self.index_path))

                return index

            logger.info("Loading existing FAISS index")

            return faiss.read_index(str(self.index_path))

        except Exception as e:

            logger.error(f"FAISS index corrupted, rebuilding: {e}")

            if self.index_path.exists():
                self.index_path.unlink()

            index = self._create_index()

            faiss.write_index(index, str(self.index_path))

            return index

    # ------------------------------------------------
    # Update Index
    # ------------------------------------------------

    def update_index(self, prototypes):

        prototypes = np.asarray(prototypes).astype("float32")

        # Normalize vectors
        faiss.normalize_L2(prototypes)

        lock = FileLock(str(self.index_path) + ".lock")

        with lock:

            if not self.index_path.exists():

                index = self._create_index()

            else:

                index = faiss.read_index(str(self.index_path))

            start_id = index.ntotal

            logger.info(f"Current vector count: {start_id}")

            vector_ids = np.arange(start_id, start_id + len(prototypes))

            index.add_with_ids(prototypes, vector_ids)

            faiss.write_index(index, str(self.index_path))

            logger.info(f"FAISS index updated → total vectors: {index.ntotal}")
            self.index = index
        return vector_ids.tolist()

    # ------------------------------------------------
    # Duplicate Check
    # ------------------------------------------------

    def check_duplicate_IM(self, prototypes, threshold=0.90, match_ratio=0.70):

        if not self.index_path.exists():
            return False

        index = faiss.read_index(str(self.index_path))

        if index.ntotal == 0:
            return False

        prototypes = np.asarray(prototypes).astype("float32")

        faiss.normalize_L2(prototypes)
        # D, I = index.search(prototypes, 5)

        # similarities = D.flatten()
        D, I = index.search(prototypes, 1)

        similarities = D[:,0]
        vector_ids = I[:, 0]

        logger.info(f"Prototype similarities: {similarities}")

    # -----------------------------
    # Filter valid matches
    # -----------------------------
        valid_mask = (similarities > threshold) & (vector_ids != -1)

        if valid_mask.sum() == 0:
            return False, None

        matched_vids = vector_ids[valid_mask].astype(int).tolist()

        # -----------------------------
        # Map vector_id → ad_id
        # -----------------------------
        vid_to_ad = self.db.get_ads_by_vector_ids(matched_vids)

        matched_ads = [
            vid_to_ad.get(vid)
            for vid in matched_vids
            if vid_to_ad.get(vid) is not None
        ]

        if not matched_ads:
            return False, None

        # -----------------------------
        # Count frequency of ad_ids
        # -----------------------------
        ad_counter = Counter(matched_ads)

        best_ad_id, best_count = ad_counter.most_common(1)[0]

        ratio = best_count / len(prototypes)

        logger.info(f"Duplicate match ratio: {ratio:.2f} for ad_id={best_ad_id}")

        if ratio >= match_ratio:
            return True, best_ad_id

        return False, None

    
    def classify_ad(self, prototypes, duration, current_ad_id):

        print("\n========== CLASSIFY START ==========")

        # -----------------------------
        # 0. Index check
        # -----------------------------
        if self.index is None:
            print("Index is None")
            return {"type": "NEW"}

        if self.index.ntotal == 0:
            print("Index is empty")
            return {"type": "NEW"}

        index = self.index
        print(f"Index size: {index.ntotal}")

        # -----------------------------
        # 1. Normalize input
        # -----------------------------
        prototypes = np.asarray(prototypes).astype("float32")
        faiss.normalize_L2(prototypes)

        print(f"Num prototypes: {len(prototypes)}")

        # -----------------------------
        # 2. DUPLICATE CHECK (STRICT)
        # -----------------------------
        is_dup, dup_ad_id = self.check_duplicate_IM(prototypes)

        if is_dup:
            return {
                "type": "DUPLICATE",
                "matched_ad_id": dup_ad_id
            }

        # -----------------------------
        # 3. FAISS search (for VARIANT only)
        # -----------------------------
        D, I = index.search(prototypes, 5)

        # -----------------------------
        # 4. Collect vector IDs
        # -----------------------------
        all_vids = set()
        for ids in I:
            for vid in ids:
                if vid != -1:
                    all_vids.add(int(vid))

        if not all_vids:
            print("No matches → NEW")
            return {"type": "NEW"}

        # -----------------------------
        # 5. DB mapping
        # -----------------------------
        vid_to_ad = self.db.get_ads_by_vector_ids(list(all_vids))

        # -----------------------------
        # 6. Filter matches
        # -----------------------------
        filtered = []

        for sims, ids in zip(D, I):
            for sim, vid in zip(sims, ids):

                if vid == -1:
                    continue

                matched_ad_id = vid_to_ad.get(int(vid))
                if matched_ad_id is None:
                    continue

                if matched_ad_id == current_ad_id:
                    continue

                filtered.append((float(sim), matched_ad_id))

        if not filtered:
            print("No valid matches → NEW")
            return {"type": "NEW"}

        # -----------------------------
        # 7. Aggregate scores
        # -----------------------------
        ad_scores = {}

        for sim, ad_id in filtered:
            ad_scores.setdefault(ad_id, []).append(sim)

        # -----------------------------
        # 8. Compute scores
        # -----------------------------
        scored_ads = []
        total_prototypes = len(prototypes)
        for ad_id, sims in ad_scores.items():
            sims = np.array(sims)

            avg_sim = sims.mean()
            count = len(sims)
            coverage = count / total_prototypes
            combined_score = avg_sim * math.log(1 + count)
            scored_ads.append({
                "ad_id": ad_id,
                "avg_sim": avg_sim,
                "count": count,
                "coverage": coverage,
                "score": combined_score
            })

            print(f"Ad {ad_id}: avg={avg_sim:.4f}, count={count}, coverage={coverage:.2f}, score={combined_score:.4f}")


        scored_ads.sort(key=lambda x: x["score"], reverse=True)

        best = scored_ads[0]

        best_ad_id = best["ad_id"]
        best_score = best["avg_sim"]      # keep original similarity for threshold
        best_count = best["count"]
        best_coverage = best["coverage"]

        print("\nBEST MATCH:")
        print(f"  ad_id={best_ad_id}")
        print(f"  avg={best_score:.4f}, count={best_count}")

        # -----------------------------
        # 9. VARIANT CHECK (LOOSE)
        # -----------------------------
        VAR_THRESHOLD = 0.75
        MIN_COUNT = 2
        MIN_COVERAGE = 0.7   # at least 40% frames should agree

        print("\nDECISION CHECK:")
        print(f"  best_avg={best_score:.4f}")
        print(f"  count={best_count}")
        print(f"  coverage={best_coverage:.2f}")

        # enforce consistency
        if best_count < MIN_COUNT:
            print("→ REJECTED: not enough matching frames")
            return {"type": "NEW", "score": float(best_score)}

        if best_coverage < MIN_COVERAGE:
            print("→ REJECTED: low coverage")
            return {"type": "NEW", "score": float(best_score)}

        if best_score >= VAR_THRESHOLD:
            print("→ VARIANT (consistent match)")
            return {
                "type": "VARIANT",
                "matched_ad_id": best_ad_id,
                "score": float(best_score)
            }

        print("→ NEW")
        return {
            "type": "NEW",
            "score": float(best_score)
        }

        # -----------------------------
        # 10. NEW
        # -----------------------------
        print("→ NEW")
        return {
            "type": "NEW",
            "score": float(best_score)
        }
# def classify_ad(self, prototypes, duration, current_ad_id):

    #     # if not self.index_path.exists():
    #     #     return "NEW"

    #     # index = faiss.read_index(str(self.index_path))
    #     if self.index is None or self.index.ntotal == 0:
    #         return {"type": "NEW"}

    #     index = self.index

    #     # if index.ntotal == 0:
    #     #     return "NEW"

    #     prototypes = np.asarray(prototypes).astype("float32")
    #     faiss.normalize_L2(prototypes)

    #     # -----------------------------
    # # Search similar vectors
    # # -----------------------------
    #     D, I = index.search(prototypes, 5)
    #     all_vids = set()
    #     for ids in I:
    #         for vid in ids:
    #             if vid != -1:
    #                 all_vids.add(vid)

    #     # vid_to_ad = self.db.get_ad_by_vector_id(list(all_vids))
    #     vid_to_ad = self.db.get_ads_by_vector_ids(list(all_vids))

    # # {vid: ad_id}
    #     # ----------------------------- 
    #     # Filter self matches
    #     # -----------------------------
    #     filtered = []

    #     for sims, ids in zip(D, I):
    #         for sim, vid in zip(sims, ids):

    #             if vid == -1:
    #                 continue

    #             matched_ad_id = vid_to_ad.get(vid)

    #             if matched_ad_id is None:
    #                 continue

    #             # 🔴 CRITICAL: ignore self
    #             if matched_ad_id == current_ad_id:
    #                 continue

    #             filtered.append((sim, matched_ad_id))

    #     # -----------------------------
    #     # No valid matches → NEW
    #     # -----------------------------
    #     if len(filtered) == 0:
    #         return {"type": "NEW"}

    #     # -----------------------------
    #     # Aggregate similarity per ad
    #     # -----------------------------
    #     ad_scores = {}

    #     for sim, ad_id in filtered:
    #         ad_scores.setdefault(ad_id, []).append(sim)
    #         # if ad_id not in ad_scores:
    #         #     ad_scores[ad_id] = []
    #         # ad_scores[ad_id].append(sim)

    #     # -----------------------------
    #     # Find best matching ad
    #     # -----------------------------
    #     best_ad_id = None
    #     best_score = 0

    #     for ad_id, sims in ad_scores.items():
    #         avg_sim = sum(sims) / len(sims)

    #         if avg_sim > best_score:
    #             best_score = avg_sim
    #             best_ad_id = ad_id

    #     # -----------------------------
    #     # Decision logic
    #     # -----------------------------
    #     # Tune these thresholds later
    #     DUP_THRESHOLD = 0.95
    #     VAR_THRESHOLD = 0.78

    #     if best_score >= DUP_THRESHOLD:
    #         return {
    #             "type": "DUPLICATE",
    #             "matched_ad_id": best_ad_id,
    #             "score": float(best_score)
    #         }

    #     elif best_score > VAR_THRESHOLD:
    #         return {
    #             "type": "VARIANT",
    #             "matched_ad_id": best_ad_id,
    #             "score": float(best_score)
    #         }

    #     else:
    #         return {"type": "NEW"}