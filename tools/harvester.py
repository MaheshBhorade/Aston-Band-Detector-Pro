import cv2
import os
import yaml
import sys
import numpy as np

# Add parent dir to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def update_config(roi_ratios, config_path="../config/settings.yaml"):
    if not os.path.exists(config_path):
        config_path = "config/settings.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['roi']['x_start'] = float(round(roi_ratios[0], 4))
    config['roi']['y_start'] = float(round(roi_ratios[1], 4))
    config['roi']['x_end'] = float(round(roi_ratios[2], 4))
    config['roi']['y_end'] = float(round(roi_ratios[3], 4))
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Updated configuration at {config_path}")

def get_display_scale(frame, max_h=720):
    h, w = frame.shape[:2]
    scale = 1.0
    if h > max_h:
        scale = max_h / h
    return scale

def main():
    video_path = input("Enter path to video file: ").strip('"')
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1. SEEKING TO AD
    print("\n--- STEP 1: FIND THE AD ---")
    print("Use the slider to find a frame where the Aston Band is fully visible.")
    print("Press 'ENTER' when you have found the frame.")
    
    cv2.namedWindow("Seek to Ad", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Pos", "Seek to Ad", 0, total_frames, lambda x: None)
    
    current_frame = None
    while True:
        pos = cv2.getTrackbarPos("Pos", "Seek to Ad")
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, current_frame = cap.read()
        if not ret: break
        
        scale = get_display_scale(current_frame)
        disp = cv2.resize(current_frame, (int(current_frame.shape[1]*scale), int(current_frame.shape[0]*scale)))
        cv2.imshow("Seek to Ad", disp)
        
        if cv2.waitKey(1) & 0xFF == 13: # ENTER
            break
            
    cv2.destroyWindow("Seek to Ad")

    # 2. ROI SELECTION (SNIPPING)
    print("\n--- STEP 2: SNIP ROI ---")
    print("Drag your mouse to select the ASTON BAND area.")
    print("Press ENTER to confirm.")
    
    # We must use the scaled version for selection but map back to original
    scale = get_display_scale(current_frame)
    disp_h, disp_w = int(current_frame.shape[0]*scale), int(current_frame.shape[1]*scale)
    disp_selection = cv2.resize(current_frame, (disp_w, disp_h))
    
    roi = cv2.selectROI("Snip Aston Band", disp_selection, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Snip Aston Band")
    
    if roi == (0, 0, 0, 0):
        print("Cancelled.")
        return

    # Map back to original resolution
    rx, ry, rw, rh = [int(v / scale) for v in roi]
    H, W = current_frame.shape[:2]
    roi_ratios = (rx/W, ry/H, (rx+rw)/W, (ry+rh)/H)
    update_config(roi_ratios)

    # 3. HARVESTING
    print("\n--- STEP 3: HARVEST SAMPLES ---")
    print(" - Press and HOLD 'S' to extract every frame of the area.")
    print(" - Press 'SPACE' to pause.")
    print(" - Press 'Q' to finish.")
    
    output_dir = "../prototypes"
    if not os.path.exists(output_dir): output_dir = "prototypes"
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    paused = False
    
    # Reset video to current position
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret: break
            
        disp = cv2.resize(frame, (disp_w, disp_h))
        cv2.rectangle(disp, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
        cv2.imshow("Harvesting - Hold 'S' to capture sequence", disp)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save original resolution crop
            crop = frame[ry:ry+rh, rx:rx+rw]
            save_path = os.path.join(output_dir, f"proto_{count:05d}.jpg")
            cv2.imwrite(save_path, crop)
            count += 1
            if count % 10 == 0: print(f"Harvested {count} frames...")
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSuccess! Harvested {count} high-res prototypes in {output_dir}")

if __name__ == "__main__":
    main()
