from pathlib import Path
import sys

import cv2
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.two_stage_detector import TwoStageAstonDetector


def main():
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    detector = TwoStageAstonDetector(config)
    unknown_dir = Path("output/unknown_aston/test_video")

    for image_path in sorted(unknown_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        features = detector.embed_crops([image])
        clip_results = detector.ad_bank.search_features(features, top_k=7)[0]
        visual_results = detector.ad_bank.search_visual([image], top_k=7)[0]

        print(f"\n{image_path.name}")
        print("  CLIP")
        for result in clip_results:
            print(f"    {result['label']:<10} {result['score']:.4f}")
        print("  VISUAL")
        for result in visual_results:
            print(f"  {result['label']:<10} {result['score']:.4f}")


if __name__ == "__main__":
    main()
