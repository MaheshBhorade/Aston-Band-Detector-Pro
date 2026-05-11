import cv2
import yaml
import os
import csv
import time
import datetime
from tqdm import tqdm
from src.detector import AstonClipDetector
from src.utils import export_to_csv

def load_config(path="config/settings.yaml"):
    if not os.path.exists(path): path = "config/settings.yaml"
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def export_debug_matches(debug_matches, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "time", "name", "score", "status"])
        for row in debug_matches:
            timestamp = row["timestamp"]
            writer.writerow([
                round(timestamp, 3),
                str(datetime.timedelta(seconds=int(timestamp))),
                row["name"],
                round(row["score"], 4),
                row["status"],
            ])
    print(f"Debug matches saved to {output_path}")

def main():
    config = load_config()
    detector = AstonClipDetector(config)
    
    input_dir = config['paths']['input_dir']
    output_dir = config['paths']['output_dir']
    snapshot_dir = os.path.join(output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.ts'))]
    
    if not video_files:
        print(f"No videos found in {input_dir}")
        return

    # --- INDUSTRIAL SPEED SETTINGS ---
    batch_size = 64 # High-speed GPU batching
    show_live = config['processing'].get('show_live', False)

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        
        # ✅ Reset detector state so previous video's data doesn't bleed in
        detector.reset()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # --- SPEED SETTINGS ---
        batch_size = 64
        # Respect config setting:
        process_every_n = max(1, int(config['processing'].get('process_every_n_frames', 1)))

        print(f"\n🚀 Processing: {video_file}")
        print(f"   > Scanning every {process_every_n} frames (Config-defined speed)")
        
        frame_idx = 0
        batch_frames = []
        batch_times = []
        
        pbar = tqdm(total=total_frames, desc="Processing")
        
        while cap.isOpened():
            # Fast-jump to the next second
            if frame_idx % process_every_n != 0:
                if not cap.grab(): break
                frame_idx += 1
                pbar.update(1)
                continue

            ret, frame = cap.read()
            if not ret: break
            
            timestamp = frame_idx / fps
            batch_frames.append(frame)
            batch_times.append(timestamp)
            
            if len(batch_frames) >= batch_size:
                detector.process_batch(batch_frames, batch_times)
                
                # Snapshot for any matches in this batch
                if detector.raw_detections:
                    # Look at recent detections that fall within the current batch
                    recent_matches = [d for d in detector.raw_detections if d[1] in batch_times]
                    for match_name, match_t, match_score in recent_matches:
                        # Find the frame corresponding to this timestamp
                        try:
                            idx = batch_times.index(match_t)
                            snap_frame = batch_frames[idx].copy()
                            H, W = snap_frame.shape[:2]
                            rx1, ry1 = int(config['roi']['x_start']*W), int(config['roi']['y_start']*H)
                            rx2, ry2 = int(config['roi']['x_end']*W), int(config['roi']['y_end']*H)
                            cv2.rectangle(snap_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 4)
                            label = f"{match_name} ({match_score:.2f})"
                            cv2.putText(
                                snap_frame,
                                label,
                                (rx1, ry1-15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 255, 0),
                                3,
                            )
                            cv2.imwrite(os.path.join(snapshot_dir, f"{match_name}_{int(match_t)}s.jpg"), snap_frame)
                        except (ValueError, IndexError):
                            continue

                batch_frames = []
                batch_times = []
            
            frame_idx += 1
            pbar.update(1)
            
        if batch_frames:
            detector.process_batch(batch_frames, batch_times)
            
        cap.release()
        pbar.close()
        
        print(f"Finalizing results...")
        segments = detector.get_final_segments()
        base_name = os.path.splitext(video_file)[0]
        export_to_csv(segments, os.path.join(output_dir, f"{base_name}_detections.csv"))
        export_debug_matches(detector.debug_matches, os.path.join(output_dir, f"{base_name}_debug_matches.csv"))
        
    print("\n Finished bro!")

if __name__ == "__main__":
    main()
