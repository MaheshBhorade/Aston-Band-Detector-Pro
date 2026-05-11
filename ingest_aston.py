import os
import cv2
import numpy as np
import yaml
import json
import faiss
from PIL import Image
from ingestion.prototype_builder import PrototypeBuilder
from ingestion.clip_embedder import CLIPEmbedder
from src.model import CLIPEngine

def load_config(path="config/settings.yaml"):
    if not os.path.exists(path): path = "config/settings.yaml"
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    clipped_dir = "prototypes/clipped_videos"
    
    print("Initializing CLIP and Prototype Builder (High Clarity Mode)...")
    embedder = CLIPEmbedder()
    # Force exactly 10 prototypes
    builder = PrototypeBuilder(max_prototypes=10, min_prototypes=10)
    engine = CLIPEngine(model_name=config['matching']['model_name'])
    
    index_path = "indexes/aston_ads.faiss"
    map_path = "indexes/aston_map.json"

    # Reset option
    reset = input("\nDo you want to CLEAR the existing index and start fresh? (y/n): ").lower() == 'y'
    if not reset and os.path.exists(map_path):
        print("Loading existing index...")
        with open(map_path, 'r') as f:
            engine.id_to_name = {int(k): v for k, v in json.load(f).items()}
            engine.index = faiss.read_index(index_path)

    if not os.path.exists(clipped_dir):
        print(f"Error: Folder {clipped_dir} not found.")
        return

    video_files = [f for f in os.listdir(clipped_dir) if f.lower().endswith(('.mp4', '.ts'))]
    if not video_files:
        print("No clipped videos found.")
        return

    print(f"\nFound {len(video_files)} clips to ingest.")

    for v_file in video_files:
        v_path = os.path.join(clipped_dir, v_file)
        
        print(f"\n--- Processing: {v_file} ---")
        ad_name = input(f"Enter the name of the ad in {v_file} (or Enter to skip): ").strip()
        if not ad_name:
            continue

        video_images = []
        cap = cv2.VideoCapture(v_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # Keep as BGR — CLIPEmbedder handles the BGR→RGB conversion
            video_images.append(frame)
        cap.release()

        if not video_images: continue

        print(f"  > Analyzing {len(video_images)} frames for clarity and diversity...")
        embeddings = embedder.embed_images(video_images)
        
        # Build exactly 10 sharp prototypes
        proto_embeddings, _ = builder.build_prototypes(embeddings, video_images)
        
        unique_proto_dict = {}
        for i, emb in enumerate(proto_embeddings):
            unique_proto_dict[f"{ad_name}_{i}"] = emb.reshape(1, -1).astype('float32')
        
        engine.add_to_index(unique_proto_dict)
        print(f"  > Successfully added 10 prototypes for '{ad_name}'.")

    os.makedirs("indexes", exist_ok=True)
    if engine.index:
        faiss.write_index(engine.index, index_path)
        with open(map_path, "w") as f:
            json.dump(engine.id_to_name, f, indent=4)
        print(f"\n✅ Done! Every ad now has exactly 10 high-quality prototypes.")
    else:
        print("\nNo ads were indexed.")

if __name__ == "__main__":
    main()
