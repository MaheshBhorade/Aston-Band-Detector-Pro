# File: content_detection_system/ingestion/frame_filter.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import logging

logger = logging.getLogger("ad_ingestion.frame_filter")


class FrameFilter:

    def __init__(self,
                 blur_threshold=40,
                 brightness_threshold=40):
        self.min_frames = 5
        self.blur_threshold = blur_threshold
        self.brightness_threshold = brightness_threshold

    # ------------------------------------------------
    # Main Filter Function
    # ------------------------------------------------

    def filter_frames(self, frames):
        fps = 1

        total_frames = len(frames)
        video_duration = total_frames / fps if fps > 0 else 0

        logger.info(f"Filtering {total_frames} frames | duration={video_duration:.2f}s")

        # ------------------------------------------------
        # Forced indices (start + end frames)
        # ------------------------------------------------
        forced_indices = set()

        if video_duration >= 10:
            for i in range(total_frames):
                t = i / fps
                if t <= 4 or t >= (video_duration - 4):
                    forced_indices.add(i)

        logger.info(f"Forced frame indices count: {len(forced_indices)}")

        filtered_frames = []
        scored_frames = []
        filtered_indices = set()

        # Stats tracking
        passed_strict = 0
        passed_forced = 0
        rejected = 0

        blur_values = []
        brightness_values = []
        entropy_values = []
        black_ratios = []

        for idx, frame in enumerate(frames):

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape

                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                brightness = np.mean(gray)

                hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
                hist = hist / (hist.sum() + 1e-9)
                entropy = -(hist * np.log2(hist + 1e-9)).sum()

                black_ratio = np.sum(gray < 10) / (h * w)

                # Track stats
                blur_values.append(blur_score)
                brightness_values.append(brightness)
                entropy_values.append(entropy)
                black_ratios.append(black_ratio)

                # --- strict filtering ---
                is_good = (
                    blur_score >= self.blur_threshold and
                    brightness >= self.brightness_threshold and
                    entropy >= 3 and
                    black_ratio <= 0.80
                )

                # --- final decision ---
                if is_good:
                    filtered_frames.append(frame)
                    filtered_indices.add(idx)
                    passed_strict += 1

                elif idx in forced_indices and black_ratio < 0.95:
                    filtered_frames.append(frame)
                    filtered_indices.add(idx)
                    passed_forced += 1

                else:
                    rejected += 1

                # --- scoring ---
                score = (
                    0.5 * blur_score +
                    0.3 * entropy +
                    0.2 * brightness -
                    50 * black_ratio
                )

                scored_frames.append((score, idx))

            except Exception as e:
                logger.warning(f"Frame filtering error at idx={idx}: {e}")

        # ------------------------------------------------
        # Log stats
        # ------------------------------------------------
        if blur_values:
            logger.info(
                f"Stats | Blur(min/avg/max): {min(blur_values):.2f}/{np.mean(blur_values):.2f}/{max(blur_values):.2f} | "
                f"Brightness(min/avg/max): {min(brightness_values):.2f}/{np.mean(brightness_values):.2f}/{max(brightness_values):.2f} | "
                f"Entropy(avg): {np.mean(entropy_values):.2f} | "
                f"BlackRatio(avg): {np.mean(black_ratios):.2f}"
            )

        logger.info(
            f"Filter summary | strict_pass={passed_strict}, forced_pass={passed_forced}, rejected={rejected}"
        )

        # ------------------------------------------------
        # Fallback
        # ------------------------------------------------
        if len(filtered_frames) < self.min_frames:

            logger.warning(
                f"Only {len(filtered_frames)} frames passed. Applying fallback to reach {self.min_frames}"
            )

            scored_frames.sort(key=lambda x: x[0], reverse=True)

            added_from_fallback = 0

            for score, idx in scored_frames:
                if len(filtered_frames) >= self.min_frames:
                    break
                if idx not in filtered_indices:
                    filtered_frames.append(frames[idx])
                    filtered_indices.add(idx)
                    added_from_fallback += 1

            logger.info(f"Fallback added {added_from_fallback} frames")

        logger.info(f"Final frames after filtering: {len(filtered_frames)}")

        return filtered_frames
    
# class FrameFilter:

#     def __init__(
#         self,
#         min_brightness=10,
#         min_entropy=3,
#         min_laplacian=20,
#         max_black_ratio=0.85,
#     ):
#         self.min_brightness = min_brightness
#         self.min_entropy = min_entropy
#         self.min_laplacian = min_laplacian
#         self.max_black_ratio = max_black_ratio

#     # ------------------------------------------------
#     # Main Filtering Function
#     # ------------------------------------------------

#     def filter_frames(self, frame_paths):

#         logger.info(f"Filtering {len(frame_paths)} frames")

#         good_frames = []

#         for path in frame_paths:

#             try:
#                 frame = cv2.imread(path)

#                 if frame is None:
#                     continue

#                 if not self._is_bad_frame(frame):
#                     good_frames.append(path)

#             except Exception as e:
#                 logger.warning(f"Frame filtering error: {path} | {e}")

#         logger.info(f"Frames after filtering: {len(good_frames)}")

#         return good_frames

#     # ------------------------------------------------
#     # Frame Quality Evaluation
#     # ------------------------------------------------

#     def _is_bad_frame(self, img):

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         h, w = gray.shape

#         # brightness
#         brightness = float(gray.mean())

#         if brightness < self.min_brightness:
#             return True

#         # entropy
#         hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
#         hist = hist / (hist.sum() + 1e-9)

#         entropy = -(hist * np.log2(hist + 1e-9)).sum()

#         if entropy < self.min_entropy:
#             return True

#         # blur detection
#         lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

#         if lap_var < self.min_laplacian:
#             return True

#         # black ratio
#         black_mask = (gray < 10).astype("uint8")
#         black_ratio = black_mask.sum() / (h * w)

#         if black_ratio > self.max_black_ratio:
#             return True

#         return False