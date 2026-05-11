import cv2
import os

def hms_to_seconds(hms):
    h, m, s = map(int, hms.split(':'))
    return h * 3600 + m * 60 + s

def main():
    video_path = r"D:\Clip\videos\source\1_TS-10_1011_202604191500_202604191600.mp4"
    output_dir = r"C:\Users\Mahesh\Downloads\L_ad\aston_clip_detector\prototypes\clipped_videos"
    os.makedirs(output_dir, exist_ok=True)

    # Your identified ROI and Segments
    roi = [2, 466, 713, 80] # [x, y, w, h]
    segments = [
        ("00:46:28", "00:46:32"),
        ("00:47:35", "00:47:40"),
        ("00:50:27", "00:50:30"),
        ("00:51:39", "00:51:43"),
        ("00:53:15", "00:53:19"),
        ("00:54:37", "00:54:41"),
        ("00:56:10", "00:56:13")
    ]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 100: fps = 25
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rx, ry, rw, rh = roi

    print(f"Processing {len(segments)} segments...")

    for i, (start_hms, end_hms) in enumerate(segments):
        start_sec = hms_to_seconds(start_hms)
        end_sec = hms_to_seconds(end_hms)
        
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        output_file = os.path.join(output_dir, f"clip_{i:03d}.mp4")
        out = cv2.VideoWriter(output_file, fourcc, fps, (rw, rh))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for f_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret: break
            
            # Crop to ROI
            crop = frame[ry:ry+rh, rx:rx+rw]
            out.write(crop)
            
        out.release()
        print(f"Successfully saved: clip_{i:03d}.mp4 ({start_hms} to {end_hms})")

    cap.release()
    print("\n✅ All 7 clips have been saved successfully!")

if __name__ == "__main__":
    main()
