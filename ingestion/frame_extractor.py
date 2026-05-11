import cv2
import logging
from pathlib import Path
import os
os.environ["OMP_NUM_THREADS"] = "1"

logger = logging.getLogger("ad_ingestion.frame_extractor")


class FrameExtractor:

    def __init__(
        self,
        output_dir="storage/prototypes",
        sample_rate_sec=1,
        resize_width=320
    ):
        """
        sample_rate_sec : extract 1 frame every N seconds
        resize_width    : resize frame for faster embedding
        """

        self.output_dir = Path(output_dir)
        self.sample_rate_sec = sample_rate_sec
        self.resize_width = resize_width

    # ------------------------------------------------
    # Main Extraction Method
    # ------------------------------------------------
    @staticmethod
    def get_video_duration(video_path):
        # import cv2

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError("Failed to open video for duration")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        cap.release()

        if fps == 0:
            return 0

        duration = frame_count / fps
        return duration
    
    def extract_frames(self, video_path: str, ad_id: str):

        logger.info(f"Starting frame extraction for {video_path}")

        frames = []

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = frame_count / fps

        logger.info(
            f"Video info → FPS: {fps:.2f}, Frames: {frame_count}, Duration: {duration:.2f}s"
        )

        interval = int(fps)  # 1 frame per second
        logger.info(f"Interval info → interval: {interval}, Frames per second")
        idx = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if idx % interval == 0:

                frames.append(frame)

            idx += 1

        cap.release()

        logger.info(f"Extracted {len(frames)} frames")

        return frames
    # ------------------------------------------------
    # Resize Frame
    # ------------------------------------------------

    def _resize_frame(self, frame):

        if self.resize_width is None:
            return frame

        h, w = frame.shape[:2]

        if w <= self.resize_width:
            return frame

        ratio = self.resize_width / w
        new_h = int(h * ratio)

        frame = cv2.resize(frame, (self.resize_width, new_h))

        return frame