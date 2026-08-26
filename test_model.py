import cv2
import numpy as np
import pyttsx3
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

MODEL_PATH = "./models/pretrained_model.pt"
VIDEO_PATH = "./videos/border.mp4"
EVIDENCE_DIR = Path("./evidence")
TARGET_CLASSES = {"person", "car", "truck", "motorcycle"}

# Create evidence directory if it doesn't exist
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech

# Define Virtual Fence Polygon [(x1, y1), (x2, y2), ...]
ZONE = np.array([(100, 200), (700, 200), (700, 450), (100, 450)], np.int32)
MIN_PERSISTENT_FRAMES = 5  # Threshold to filter out single-frame false positives

if not Path(VIDEO_PATH).exists():
    raise FileNotFoundError(f"{VIDEO_PATH} not found — check the file is in videos/.")

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO_PATH} — file may be corrupted or codec unsupported.")

frame_count = 0
zone_track_history = {}  # {track_id: consecutive_frame_count}
captured_tracks = set()   # Tracks objects that already triggered an alert/evidence capture

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    
    # Create a copy of the frame to draw on for display/evidence saving
    annotated_frame = frame.copy()
    cv2.polylines(annotated_frame, [ZONE], isClosed=True, color=(0, 255, 255), thickness=2)

    results = model.track(annotated_frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    for result in results:
        boxes = result.boxes
        if boxes.id is None:
            continue

        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = result.names[class_id]

            if label in TARGET_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Check if center point is inside the restricted polygon
                is_inside = cv2.pointPolygonTest(ZONE, (cx, cy), measureDist=False) >= 0

                if is_inside:
                    zone_track_history[track_id] = zone_track_history.get(track_id, 0) + 1
                    
                    # Alert threshold met
                    if zone_track_history[track_id] >= MIN_PERSISTENT_FRAMES:
                        print(f"🚨 INTRUSION ALERT | Frame {frame_count:>4} | {label:<10} | Track ID {track_id:<4} | conf {confidence:.2f}")

                        # Trigger Voice Alert & Save Evidence once per track ID
                        if track_id not in captured_tracks:
                            # 1. Audio Voice Announcement
                            tts_engine.say(f"Alert! {label} detected in restricted zone.")
                            tts_engine.runAndWait()

                            # 2. Save Evidence Image
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = EVIDENCE_DIR / f"intrusion_ID{track_id}_{label}_{timestamp}.jpg"
                            
                            # Draw bounding box & centroid on evidence frame
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                            
                            cv2.imwrite(str(filename), annotated_frame)
                            print(f"📸 Evidence saved to {filename}")
                            
                            captured_tracks.add(track_id)
                else:
                    # Reset counter if object leaves the zone
                    zone_track_history[track_id] = 0

cap.release()
print(f"\nDone. Processed {frame_count} frames.")