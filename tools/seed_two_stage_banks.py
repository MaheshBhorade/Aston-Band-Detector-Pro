import os
import argparse
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".ts"}
CURRENT_CLIP_LABELS = {
    "clip_000": "cake",
    "clip_001": "perfume",
    "clip_002": "mek",
    "clip_003": "berry",
    "clip_004": "lameron",
    "clip_005": "MG Hector",
    "clip_006": "icell",
}


def extract_samples(video_path, output_dir, max_samples=10):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return 0

    sample_indices = [
        int(i * (frame_count - 1) / max(max_samples - 1, 1))
        for i in range(max_samples)
    ]

    saved = 0
    for sample_idx, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        stem = video_path.stem.replace(" ", "_")
        output_path = output_dir / f"{stem}_{sample_idx:02d}.jpg"
        cv2.imwrite(str(output_path), frame)
        saved += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-current-labels",
        action="store_true",
        help="Use the known labels for clip_000..clip_006 in this project.",
    )
    args = parser.parse_args()

    clipped_dir = Path("prototypes/clipped_videos")
    stage1_aston_dir = Path("banks/stage1/aston_band")
    stage1_ignore_dir = Path("banks/stage1/ignore")
    ads_dir = Path("banks/ads")

    if not clipped_dir.exists():
        print(f"Missing folder: {clipped_dir}")
        return

    videos = [
        path for path in sorted(clipped_dir.iterdir())
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not videos:
        print(f"No videos found in {clipped_dir}")
        return

    print("This will extract sample images into banks/stage1 and banks/ads.")
    print("For normal ad clips, enter the ad class name. Press Enter to skip ad-bank seeding.")

    for video_path in videos:
        lower_name = video_path.name.lower()
        if "ignore" in lower_name:
            saved = extract_samples(video_path, stage1_ignore_dir)
            print(f"{video_path.name}: saved {saved} IGNORE samples")
            continue

        saved_stage1 = extract_samples(video_path, stage1_aston_dir)
        print(f"\n{video_path.name}: saved {saved_stage1} ASTON_BAND samples")

        if args.use_current_labels:
            ad_name = CURRENT_CLIP_LABELS.get(video_path.stem, "")
        else:
            ad_name = input(f"Ad class name for {video_path.name}: ").strip()

        if not ad_name:
            continue

        saved_ad = extract_samples(video_path, ads_dir / ad_name)
        print(f"{video_path.name}: saved {saved_ad} samples for ad '{ad_name}'")

    print("\nBank seeding finished.")


if __name__ == "__main__":
    main()
