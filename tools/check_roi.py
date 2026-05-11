import cv2
import yaml
import os

def load_config():
    with open("config/settings.yaml", 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    roi = config['roi']
    input_dir = config['paths']['input_dir']
    
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.ts'))]
    if not video_files:
        print("No videos found in input_dir")
        return

    video_path = os.path.join(input_dir, video_files[0])
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to read frame")
        return

    H, W = frame.shape[:2]
    x1, x2 = int(roi['x_start']*W), int(roi['x_end']*W)
    y1, y2 = int(roi['y_start']*H), int(roi['y_end']*H)

    # Draw ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(frame, "DETECTION ROI", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    output_path = "roi_verification.jpg"
    cv2.imwrite(output_path, frame)
    print(f"\n✅ ROI verification image saved to: {os.path.abspath(output_path)}")
    print(f"   Please open this image and check if the green box covers the Aston band correctly.")

if __name__ == "__main__":
    main()
