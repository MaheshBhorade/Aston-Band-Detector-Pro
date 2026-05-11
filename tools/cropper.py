import cv2
import os
import yaml
import sys
import time
import argparse

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_config(path="config/settings.yaml"):
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def update_config_roi(ratios, config_path="config/settings.yaml"):
    config = load_config(config_path)
    config['roi']['x_start'] = float(round(ratios[0], 4))
    config['roi']['y_start'] = float(round(ratios[1], 4))
    config['roi']['x_end'] = float(round(ratios[2], 4))
    config['roi']['y_end'] = float(round(ratios[3], 4))
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config updated with new ROI: {ratios}")

def get_display_scale(frame, max_h=720):
    h, w = frame.shape[:2]
    return max_h / h if h > max_h else 1.0

def roi_from_config(config, width, height):
    roi_config = config['roi']
    x1 = int(roi_config['x_start'] * width)
    y1 = int(roi_config['y_start'] * height)
    x2 = int(roi_config['x_end'] * width)
    y2 = int(roi_config['y_end'] * height)
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]

def next_output_path(output_dir, prefix):
    existing = []
    for name in os.listdir(output_dir):
        if name.lower().startswith(prefix.lower()) and name.lower().endswith(".mp4"):
            stem = os.path.splitext(name)[0]
            suffix = stem[len(prefix):].lstrip("_")
            if suffix.isdigit():
                existing.append(int(suffix))

    next_idx = max(existing, default=0) + 1
    return os.path.join(output_dir, f"{prefix}_{next_idx:03d}.mp4")

def save_sample_images(video_file, sample_dir, max_samples=10):
    os.makedirs(sample_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return 0

    saved = 0
    stem = os.path.splitext(os.path.basename(video_file))[0]
    for i in range(max_samples):
        frame_idx = int(i * (frame_count - 1) / max(max_samples - 1, 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        output_path = os.path.join(sample_dir, f"{stem}_{i:02d}.jpg")
        cv2.imwrite(output_path, frame)
        saved += 1

    cap.release()
    return saved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to the video file.")
    parser.add_argument("--prefix", default="Ignore", help="Output clip prefix, e.g. Ignore or clip.")
    parser.add_argument("--manual-roi", action="store_true", help="Select ROI manually instead of using config ROI.")
    parser.add_argument("--seed-ignore-bank", action="store_true", help="Save extracted clip frames to banks/stage1/ignore.")
    args = parser.parse_args()

    video_path = args.video or input("Enter path to large video file: ").strip('"')
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    config = load_config()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 100: fps = 25
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"\nVideo loaded: {W}x{H} @ {fps:.2f}fps")
    print("\n--- SURGICAL CROPPER ---")
    print("1. Use the slider to find an ignore/announcement segment.")
    print("2. Press '[' to mark the START.")
    print("3. Press ']' to mark the END.")
    print("4. Press 'Q' to finish and export.")
    print("Keys: SPACE play/pause | A back 1s | D forward 1s")
    
    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Pos", "Video Player", 0, total_frames, lambda x: None)
    
    segments = []
    current_start = None
    playing = False
    current_pos = 0
    roi = None if args.manual_roi else roi_from_config(config, W, H)
    scale = 1.0

    if roi:
        print(f"Using fixed ROI from config: {roi}")

    while True:
        if playing:
            current_pos += 1
            if current_pos >= total_frames:
                playing = False
                current_pos = total_frames - 1
            cv2.setTrackbarPos("Pos", "Video Player", current_pos)
        else:
            current_pos = cv2.getTrackbarPos("Pos", "Video Player")

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        ret, frame = cap.read()
        if not ret: break
        
        scale = get_display_scale(frame)
        disp_w, disp_h = int(W*scale), int(H*scale)
        disp = cv2.resize(frame, (disp_w, disp_h))
        
        # UI
        timestamp = time.strftime('%H:%M:%S', time.gmtime(current_pos / fps))
        cv2.putText(disp, f"Time: {timestamp}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if roi:
            rx, ry, rw, rh = [int(v * scale) for v in [roi[0], roi[1], roi[2], roi[3]]]
            cv2.rectangle(disp, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)

        if current_start is not None:
            cv2.putText(disp, ">>> START MARKED <<<", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Video Player", disp)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
        elif key == ord('['):
            current_start = current_pos
            if roi is None:
                print("\n[!] Please select the Aston Band area in the new window...")
                sel_roi = cv2.selectROI("Snip ROI", disp, False)
                cv2.destroyWindow("Snip ROI")
                if sel_roi != (0, 0, 0, 0):
                    roi = [int(v / scale) for v in sel_roi]
                    # Boundary check
                    roi[0] = max(0, min(roi[0], W-1))
                    roi[1] = max(0, min(roi[1], H-1))
                    roi[2] = min(roi[2], W - roi[0])
                    roi[3] = min(roi[3], H - roi[1])
                    print(f"ROI locked: {roi}")
                    # Save ratios to config
                    ratios = [roi[0]/W, roi[1]/H, (roi[0]+roi[2])/W, (roi[1]+roi[3])/H]
                    update_config_roi(ratios)
            print(f"Start marked at {timestamp}")
        elif key == ord(']'):
            if current_start is not None:
                segments.append((current_start, current_pos))
                print(f"End marked at {timestamp}")
                current_start = None
        elif key == ord('d'):
            current_pos = min(total_frames - 1, current_pos + int(fps))
            cv2.setTrackbarPos("Pos", "Video Player", current_pos)
        elif key == ord('a'):
            current_pos = max(0, current_pos - int(fps))
            cv2.setTrackbarPos("Pos", "Video Player", current_pos)
        
    cv2.destroyAllWindows()

    if not segments or not roi:
        print("No segments marked or no ROI selected.")
        cap.release()
        return

    output_dir = os.path.join(PROJECT_ROOT, "prototypes", "clipped_videos")
    os.makedirs(output_dir, exist_ok=True)
    rx, ry, rw, rh = roi
    
    print(f"\n[STEP 3] Saving {len(segments)} segments to {output_dir}...")
    
    # We use OpenCV VideoWriter to avoid FFmpeg encoder issues
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Standard MP4 codec
    
    saved_files = []
    for start_f, end_f in segments:
        output_file = next_output_path(output_dir, args.prefix)
        saved_files.append(output_file)
        out = cv2.VideoWriter(output_file, fourcc, fps, (rw, rh))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for f_idx in range(start_f, end_f + 1):
            ret, frame = cap.read()
            if not ret: break
            
            # Crop to ROI
            crop = frame[ry:ry+rh, rx:rx+rw]
            out.write(crop)
            
        out.release()
        print(f"Saved: {os.path.basename(output_file)}")

    if args.seed_ignore_bank:
        sample_dir = os.path.join(PROJECT_ROOT, "banks", "stage1", "ignore")
        total_saved = 0
        for output_file in saved_files:
            total_saved += save_sample_images(output_file, sample_dir)
        print(f"Saved {total_saved} ignore sample images to {sample_dir}")

    cap.release()
    print("\nAll videos saved successfully.")

if __name__ == "__main__":
    main()
