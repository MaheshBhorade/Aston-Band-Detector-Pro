import os
import cv2
import json
import faiss
import torch
from PIL import Image
from .model import CLIPEngine


class AstonClipDetector:
    def __init__(self, config):
        self.config = config
        self.engine = CLIPEngine(model_name=config["matching"]["model_name"])
        self.load_faiss_index()

        matching_config = config["matching"]
        self.start_threshold = matching_config["threshold"]
        self.continue_threshold = matching_config.get(
            "continue_threshold",
            self.start_threshold - matching_config.get("continue_threshold_delta", 0.08),
        )
        self.confirmation_frames = max(1, int(matching_config.get("confirmation_frames", 1)))
        self.gap_threshold = float(matching_config.get("gap_threshold", 5.0))

        self.roi_config = config["roi"]
        self.raw_detections = []
        self.debug_matches = []
        self.active_ad = None
        self.pending_ad = None
        self.pending_hits = []

    def load_faiss_index(self):
        index_path = "indexes/aston_ads.faiss"
        map_path = "indexes/aston_map.json"
        if os.path.exists(index_path) and os.path.exists(map_path):
            self.engine.index = faiss.read_index(index_path)
            with open(map_path, "r") as f:
                mapping = json.load(f)
                self.engine.id_to_name = {int(k): v for k, v in mapping.items()}
            print(f"Loaded Faiss index with {len(self.engine.id_to_name)} prototypes.")
        else:
            print("No Faiss index found.")

    def reset(self):
        """Clears the detector state for a new video."""
        self.raw_detections = []
        self.debug_matches = []
        self.active_ad = None
        self.pending_ad = None
        self.pending_hits = []
        print("Detector state reset.")

    def _reset_pending(self):
        self.pending_ad = None
        self.pending_hits = []

    def _record_debug(self, timestamp, name, score, status):
        self.debug_matches.append(
            {
                "timestamp": timestamp,
                "name": name,
                "score": score,
                "status": status,
            }
        )

    def _confirm_or_queue_hit(self, clean_name, timestamp, score):
        if self.confirmation_frames <= 1:
            self.active_ad = clean_name
            self.raw_detections.append((clean_name, timestamp, score))
            return "hit"

        if self.pending_ad != clean_name:
            self.pending_ad = clean_name
            self.pending_hits = []

        self.pending_hits.append((clean_name, timestamp, score))

        if len(self.pending_hits) >= self.confirmation_frames:
            self.active_ad = clean_name
            self.raw_detections.extend(self.pending_hits)
            self._reset_pending()
            return "confirmed"

        return "pending"

    def get_roi_crop(self, frame):
        h, w = frame.shape[:2]
        y1 = int(self.roi_config["y_start"] * h)
        y2 = int(self.roi_config["y_end"] * h)
        x1 = int(self.roi_config["x_start"] * w)
        x2 = int(self.roi_config["x_end"] * w)
        return frame[y1:y2, x1:x2]

    def process_batch(self, frame_batch, timestamps):
        if self.engine.index is None or not frame_batch:
            return

        crops = []
        for frame in frame_batch:
            crop = self.get_roi_crop(frame)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(rgb))

        inputs = torch.stack([self.engine.preprocess(img) for img in crops]).to(self.engine.device)

        with torch.no_grad():
            embeddings = self.engine.model.encode_image(inputs)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            feats = embeddings.cpu().numpy().astype("float32")

        faiss.normalize_L2(feats)
        scores, indices = self.engine.index.search(feats, 1)

        for i in range(len(indices)):
            score = float(scores[i][0])
            idx = indices[i][0]
            timestamp = timestamps[i]

            if idx == -1:
                self._reset_pending()
                self._record_debug(timestamp, "", score, "no_match")
                continue

            name = self.engine.id_to_name[idx]
            clean_name = name.rsplit("_", 1)[0]

            if clean_name.upper().startswith("IGNORE"):
                self.active_ad = None
                self._reset_pending()
                self._record_debug(timestamp, clean_name, score, "ignore")
                continue

            if self.active_ad == clean_name:
                if score >= self.continue_threshold:
                    self.raw_detections.append((clean_name, timestamp, score))
                    self._record_debug(timestamp, clean_name, score, "hit")
                else:
                    self.active_ad = None
                    self._reset_pending()
                    self._record_debug(timestamp, clean_name, score, "ended")
            elif score >= self.start_threshold:
                status = self._confirm_or_queue_hit(clean_name, timestamp, score)
                self._record_debug(timestamp, clean_name, score, status)
            else:
                self._reset_pending()
                self._record_debug(timestamp, clean_name, score, "below_threshold")

    def get_final_segments(self):
        if not self.raw_detections:
            return []

        segments = []
        self.raw_detections.sort(key=lambda x: (x[0], x[1]))

        ad_id, start, score = self.raw_detections[0]
        prev_t = start
        max_score = score

        for name, t, s in self.raw_detections[1:]:
            if name == ad_id and (t - prev_t) <= self.gap_threshold:
                prev_t = t
                max_score = max(max_score, s)
            else:
                segments.append((ad_id, start, prev_t, max_score))
                ad_id, start, score = name, t, s
                prev_t = t
                max_score = s
        segments.append((ad_id, start, prev_t, max_score))

        min_duration = self.config["matching"].get("min_duration", 2.0)
        final_list = [m for m in segments if (m[2] - m[1]) >= min_duration]
        final_list = self._remove_overlapping_segments(final_list)
        final_list.sort(key=lambda x: x[1])

        return final_list

    def _remove_overlapping_segments(self, segments):
        if len(segments) <= 1:
            return segments

        sorted_segments = sorted(
            segments,
            key=lambda x: ((x[2] - x[1]), x[3]),
            reverse=True,
        )
        kept = []

        for segment in sorted_segments:
            _, start, end, _ = segment
            duration = max(end - start, 1e-6)
            is_duplicate_window = False

            for kept_segment in kept:
                _, kept_start, kept_end, _ = kept_segment
                overlap = max(0, min(end, kept_end) - max(start, kept_start))
                if overlap / duration >= 0.70:
                    is_duplicate_window = True
                    break

            if not is_duplicate_window:
                kept.append(segment)

        return kept
