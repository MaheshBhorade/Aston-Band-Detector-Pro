import os
import argparse

import cv2
import yaml
from tqdm import tqdm

from src.two_stage_detector import TwoStageAstonDetector


def load_config(path="config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Optional single video filename from input_dir to process.")
    args = parser.parse_args()

    config = load_config()
    detector = TwoStageAstonDetector(config)

    if not detector.stage1_bank.is_ready:
        print("Stage 1 bank is empty. Add images to banks/stage1/aston_band and banks/stage1/ignore.")
        return

    input_dir = config["paths"]["input_dir"]
    output_dir = config["paths"]["output_dir"]
    video_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".mp4", ".avi", ".ts"))
    ]
    if args.video:
        video_files = [f for f in video_files if f == args.video]

    if not video_files:
        print(f"No videos found in {input_dir}")
        return

    batch_size = int(config["two_stage"].get("batch_size", 64))
    process_every_n = max(1, int(config["processing"].get("process_every_n_frames", 1)))

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        detector.reset()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nTwo-stage processing: {video_file}")
        print(f"   > scanning every {process_every_n} frames")

        frame_idx = 0
        batch_frames = []
        batch_times = []

        pbar = tqdm(total=total_frames, desc="Processing")
        while cap.isOpened():
            if frame_idx % process_every_n != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                pbar.update(1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            batch_frames.append(frame)
            batch_times.append(frame_idx / fps)

            if len(batch_frames) >= batch_size:
                detector.process_batch(batch_frames, batch_times)
                batch_frames = []
                batch_times = []

            frame_idx += 1
            pbar.update(1)

        if batch_frames:
            detector.process_batch(batch_frames, batch_times)

        detector.finish()
        cap.release()
        pbar.close()

        detector.export_outputs(video_file, output_dir)

    print("\nTwo-stage run finished.")


if __name__ == "__main__":
    main()
