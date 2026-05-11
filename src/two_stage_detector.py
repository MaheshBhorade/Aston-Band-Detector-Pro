import csv
import datetime
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from PIL import Image

from .folder_bank import FolderImageBank
from .model import CLIPEngine
from .utils import export_to_csv


class TwoStageAstonDetector:
    def __init__(self, config):
        self.config = config
        two_stage = config["two_stage"]

        self.engine = CLIPEngine(model_name=config["matching"]["model_name"])
        self.stage1_bank = FolderImageBank(self.engine, two_stage["stage1_bank_dir"])
        self.ad_bank = FolderImageBank(self.engine, two_stage["ad_bank_dir"])
        self.stage1_bank.load()
        self.ad_bank.load()

        self.stage1_threshold = float(two_stage.get("stage1_threshold", 0.82))
        self.stage1_margin = float(two_stage.get("stage1_margin", 0.03))
        self.save_debug_csv = bool(two_stage.get("save_debug_csv", False))
        self.ad_threshold = float(two_stage.get("ad_threshold", 0.88))
        self.ad_margin = float(two_stage.get("ad_margin", 0.03))
        self.ad_segment_margin = float(two_stage.get("ad_segment_margin", 0.015))
        self.visual_threshold = float(two_stage.get("visual_threshold", 0.86))
        self.visual_segment_margin = float(two_stage.get("visual_segment_margin", 0.03))
        self.min_vote_ratio = float(two_stage.get("min_vote_ratio", 0.55))
        self.confirmation_frames = max(1, int(two_stage.get("confirmation_frames", 3)))
        self.end_gap_seconds = float(two_stage.get("end_gap_seconds", 1.2))
        self.min_duration = float(two_stage.get("min_duration", 1.2))
        self.max_segment_frames = max(1, int(two_stage.get("max_segment_frames", 12)))

        self.roi_config = config["roi"]
        self.reset()

    def reset(self):
        self.segments = []
        self.debug_rows = []
        self.active = False
        self.pending_hits = []
        self.current_frames = []
        self.current_times = []
        self.last_hit_time = None

    def get_roi_crop(self, frame):
        h, w = frame.shape[:2]
        y1 = int(self.roi_config["y_start"] * h)
        y2 = int(self.roi_config["y_end"] * h)
        x1 = int(self.roi_config["x_start"] * w)
        x2 = int(self.roi_config["x_end"] * w)
        return frame[y1:y2, x1:x2]

    def embed_crops(self, crops):
        images = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))

        with torch.no_grad():
            inputs = torch.stack([self.engine.preprocess(img) for img in images]).to(self.engine.device)
            embeddings = self.engine.model.encode_image(inputs)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

        features = embeddings.cpu().numpy().astype("float32")
        faiss.normalize_L2(features)
        return features

    def process_batch(self, frames, timestamps):
        if not frames:
            return

        crops = [self.get_roi_crop(frame) for frame in frames]
        features = self.embed_crops(crops)
        stage1_results = self.stage1_bank.search_features(features, top_k=2)

        for crop, timestamp, result in zip(crops, timestamps, stage1_results):
            is_aston, label, score, margin = self._is_aston_band(result)
            self.debug_rows.append(
                {
                    "timestamp": timestamp,
                    "stage1_label": label,
                    "stage1_score": score,
                    "stage1_margin": margin,
                    "decision": "ASTON_BAND" if is_aston else "IGNORE",
                }
            )

            if is_aston:
                self._accept_aston_frame(crop, timestamp)
            else:
                self._maybe_close_segment(timestamp)

    def _is_aston_band(self, result):
        if not result:
            return False, "", 0.0, 0.0

        best = result[0]
        second_score = result[1]["score"] if len(result) > 1 else 0.0
        label = best["label"]
        score = best["score"]
        margin = score - second_score

        is_aston = (
            label.lower() == "aston_band"
            and score >= self.stage1_threshold
            and margin >= self.stage1_margin
        )
        return is_aston, label, score, margin

    def _accept_aston_frame(self, crop, timestamp):
        self.last_hit_time = timestamp

        if not self.active:
            self.pending_hits.append((crop, timestamp))
            if len(self.pending_hits) < self.confirmation_frames:
                return

            self.active = True
            for pending_crop, pending_time in self.pending_hits:
                self.current_frames.append(pending_crop)
                self.current_times.append(pending_time)
            self.pending_hits = []
            return

        self.current_frames.append(crop)
        self.current_times.append(timestamp)

    def _maybe_close_segment(self, timestamp):
        self.pending_hits = []
        if not self.active or self.last_hit_time is None:
            return

        if timestamp - self.last_hit_time >= self.end_gap_seconds:
            self._close_current_segment()

    def finish(self):
        if self.active:
            self._close_current_segment()

    def _close_current_segment(self):
        if not self.current_frames:
            self._clear_active_segment()
            return

        start = self.current_times[0]
        end = self.current_times[-1]
        if end - start >= self.min_duration:
            selected_frames = self._select_segment_frames(self.current_frames)
            label, confidence, votes = self.identify_ad(selected_frames)
            self.segments.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "confidence": confidence,
                    "votes": votes,
                    "frames": selected_frames,
                }
            )

        self._clear_active_segment()

    def _clear_active_segment(self):
        self.active = False
        self.current_frames = []
        self.current_times = []
        self.last_hit_time = None

    def _select_segment_frames(self, frames):
        if len(frames) <= self.max_segment_frames:
            return frames

        scored = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
            scored.append((clarity, frame))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [frame for _, frame in scored[:self.max_segment_frames]]

    def identify_ad(self, crops):
        if not self.ad_bank.is_ready:
            return "UNKNOWN_ASTON", 0.0, ""

        features = self.embed_crops(crops)
        results = self.ad_bank.search_features(features, top_k=len(set(self.ad_bank.labels)))
        visual_results = self.ad_bank.search_visual(crops, top_k=len(set(self.ad_bank.labels)))

        votes = []
        score_by_label = defaultdict(list)
        for result in results:
            if not result:
                continue

            best = result[0]
            second_score = result[1]["score"] if len(result) > 1 else 0.0
            margin = best["score"] - second_score

            if best["score"] >= self.ad_threshold and margin >= self.ad_margin:
                votes.append(best["label"])
            for item in result:
                score_by_label[item["label"]].append(item["score"])

        if not score_by_label:
            return "UNKNOWN_ASTON", 0.0, ""

        class_scores = []
        for label, scores in score_by_label.items():
            sorted_scores = sorted(scores, reverse=True)
            keep_count = max(1, int(len(sorted_scores) * 0.60))
            class_scores.append((label, float(np.mean(sorted_scores[:keep_count]))))

        class_scores.sort(key=lambda item: item[1], reverse=True)
        best_label, best_score = class_scores[0]
        second_score = class_scores[1][1] if len(class_scores) > 1 else 0.0
        segment_margin = best_score - second_score

        visual_scores = self._aggregate_result_scores(visual_results)
        visual_label, visual_score, visual_margin = self._best_score_with_margin(visual_scores)

        vote_counts = Counter(votes)
        vote_ratio = vote_counts.get(best_label, 0) / len(crops)
        votes_text = self._format_votes(vote_counts)

        if visual_score >= self.visual_threshold and visual_margin >= self.visual_segment_margin:
            return visual_label, visual_score, f"visual:{visual_label}"

        if best_score < self.ad_threshold:
            return "UNKNOWN_ASTON", best_score, votes_text

        if segment_margin < self.ad_segment_margin and vote_ratio < self.min_vote_ratio:
            return "UNKNOWN_ASTON", best_score, votes_text

        return best_label, best_score, votes_text

    def _aggregate_result_scores(self, results):
        score_by_label = defaultdict(list)
        for result in results:
            for item in result:
                score_by_label[item["label"]].append(item["score"])

        class_scores = []
        for label, scores in score_by_label.items():
            sorted_scores = sorted(scores, reverse=True)
            keep_count = max(1, int(len(sorted_scores) * 0.60))
            class_scores.append((label, float(np.mean(sorted_scores[:keep_count]))))

        class_scores.sort(key=lambda item: item[1], reverse=True)
        return class_scores

    def _best_score_with_margin(self, class_scores):
        if not class_scores:
            return "", 0.0, 0.0

        label, score = class_scores[0]
        second_score = class_scores[1][1] if len(class_scores) > 1 else 0.0
        return label, score, score - second_score

    def _format_votes(self, vote_counts):
        return ";".join(f"{label}:{count}" for label, count in vote_counts.most_common())

    def export_outputs(self, video_file, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base_name = Path(video_file).stem

        detections = [
            (segment["label"], segment["start"], segment["end"], segment["confidence"])
            for segment in self.segments
            if segment["label"] != "UNKNOWN_ASTON"
        ]
        detections_path = self._safe_output_path(output_dir, f"{base_name}_two_stage_detections.csv")
        export_to_csv(detections, str(detections_path))
        self._export_segment_summary(base_name, output_dir)
        self._export_unknowns(base_name, output_dir)
        if self.save_debug_csv:
            self._export_debug(base_name, output_dir)

    def _safe_output_path(self, output_dir, filename):
        path = Path(output_dir) / filename
        if not path.exists():
            return path

        try:
            with open(path, "a"):
                return path
        except PermissionError:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")

    def _export_segment_summary(self, base_name, output_dir):
        path = self._safe_output_path(output_dir, f"{base_name}_two_stage_segments.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "start", "end", "duration", "confidence", "votes"])
            for segment in self.segments:
                start = segment["start"]
                end = segment["end"]
                writer.writerow(
                    [
                        segment["label"],
                        str(datetime.timedelta(seconds=int(start))),
                        str(datetime.timedelta(seconds=int(end))),
                        round(end - start, 2),
                        round(segment["confidence"], 4),
                        segment["votes"],
                    ]
                )
        print(f"Segment summary saved to {path}")

    def _export_unknowns(self, base_name, output_dir):
        unknown_dir = Path(output_dir) / "unknown_aston" / base_name
        unknown_dir.mkdir(parents=True, exist_ok=True)

        for idx, segment in enumerate(self.segments, 1):
            if segment["label"] != "UNKNOWN_ASTON":
                continue

            frame = segment["frames"][0]
            start = int(segment["start"])
            cv2.imwrite(str(unknown_dir / f"unknown_{idx:04d}_{start}s.jpg"), frame)

    def _export_debug(self, base_name, output_dir):
        path = Path(output_dir) / f"{base_name}_two_stage_debug.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "time", "stage1_label", "stage1_score", "stage1_margin", "decision"])
            for row in self.debug_rows:
                timestamp = row["timestamp"]
                writer.writerow(
                    [
                        round(timestamp, 3),
                        str(datetime.timedelta(seconds=int(timestamp))),
                        row["stage1_label"],
                        round(row["stage1_score"], 4),
                        round(row["stage1_margin"], 4),
                        row["decision"],
                    ]
                )
        print(f"Debug saved to {path}")
