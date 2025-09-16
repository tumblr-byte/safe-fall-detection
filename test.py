import os
import cv2
from ultralytics import YOLO
import time

video_path = ""  # add your video path
cap = cv2.VideoCapture(video_path)
output_path = "/content/output.mp4"
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

model = YOLO("/content/best.pt")
classes = ["Fall Detected", "Walking", "Sitting"]

fall_start_time = None  # initialize timer

def detect_action(frame):
    results = model(frame)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls_id = box.cls[0]
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return cls_id, conf, x1, y1, x2, y2
    return None, None, None, None, None, None

while True:
    ret, frame = cap.read()
    if not ret:
        print("There is an error reading the frame")
        break

    cls_id, conf, x1, y1, x2, y2 = detect_action(frame)
    if cls_id is None:
        out.write(frame)
        continue

    label = classes[int(cls_id)]

    if label == "Fall Detected":
        if fall_start_time is None:
            fall_start_time = time.time()
        else:
            elapsed = time.time() - fall_start_time
            if elapsed >= 5:  # 10 seconds threshold
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "ALERT! Fall detected for 5s",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
            else:
                remaining = int(10 - elapsed)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(frame, f"Fall detected, alert in {remaining}s",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)
    else:
        fall_start_time = None
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 123, 3), 3)
        cv2.putText(frame, label,
                    (x1 - 10, y1),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
