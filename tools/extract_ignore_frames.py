import argparse
import os
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".ts", ".mov", ".mkv"}


def extract_frames(video_path, output_dir, samples_per_video):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        print(f"Skipped {video_path.name}: no frames")
        return 0

    saved = 0
    for i in range(samples_per_video):
        frame_idx = int(i * (frame_count - 1) / max(samples_per_video - 1, 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        output_path = output_dir / f"{video_path.stem}_{i:02d}.jpg"
        cv2.imwrite(str(output_path), frame)
        saved += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "prototypes" / "clipped_videos"),
        help="Folder containing ignore video clips.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "banks" / "stage1" / "ignore"),
        help="Folder where extracted ignore frames will be saved.",
    )
    parser.add_argument(
        "--samples-per-video",
        type=int,
        default=10,
        help="Number of evenly spaced frames to extract from each video.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        return

    videos = [
        path for path in sorted(input_dir.iterdir())
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not videos:
        print(f"No videos found in: {input_dir}")
        return

    total = 0
    for video_path in videos:
        saved = extract_frames(video_path, output_dir, args.samples_per_video)
        total += saved
        print(f"{video_path.name}: saved {saved} frames")

    print(f"\nDone. Saved {total} ignore frames to: {output_dir}")


if __name__ == "__main__":
    main()
