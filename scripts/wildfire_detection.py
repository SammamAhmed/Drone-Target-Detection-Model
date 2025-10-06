import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv
from utils.drawing import draw_box_with_label
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    ap.add_argument("--source", default="0")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--alert-threshold", type=float, default=0.5,
                    help="Confidence threshold for fire alerts")
    ap.add_argument("--custom-model", help="Path to custom wildfire model")
    return ap.parse_args()


def main():
    args = parse_args()
    source = 0 if args.source == "0" else args.source

    # Use custom model if provided
    model_path = args.custom_model if args.custom_model else args.model
    model = YOLO(model_path)
    names = model.names

    # If using custom wildfire model, look for wildfire class
    if args.custom_model:
        print(f"[INFO] Using custom wildfire model: {model_path}")
        # Custom model should have wildfire class (class 1)
        if 'wildfire' in names.values():
            target_ids = [i for i, n in names.items() if n.lower()
                          == 'wildfire']
            print(
                f"[INFO] Wildfire detection active - class: {[names[i] for i in target_ids]}")
        else:
            print(
                f"[WARNING] Custom model doesn't have 'wildfire' class. Available classes: {list(names.values())}")
            target_ids = list(names.keys())
    else:
        # Look for fire-related classes in standard YOLO
        fire_classes = {"fire", "smoke", "flame"}
        target_ids = [i for i, n in names.items() if n.lower() in fire_classes]

        if not target_ids:
            print(
                f"[WARNING] No fire-related classes found in model. Available classes:")
            for i, name in names.items():
                print(f"  {i}: {name}")
            print(
                f"\n[INFO] You may need a custom fire detection model for proper wildfire detection.")
            print(f"[INFO] Currently detecting all classes for demonstration...")
            # For demonstration, detect all classes
            target_ids = list(names.keys())
        else:
            print(
                f"[INFO] Wildfire detection classes: {[names[i] for i in target_ids]}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open source.")
        return

    tracker = sv.ByteTrack()
    frame_idx = 0
    timing = []
    fire_alert_count = 0

    print("[INFO] Wildfire Detection System Active. ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t0 = time.time()

        # Run YOLO detection
        yres = model.predict(frame, conf=args.conf,
                             imgsz=args.imgsz, verbose=False, classes=target_ids)

        if len(yres[0].boxes) == 0:
            # No detections, just show frame with FPS
            dt = time.time() - t0
            timing.append(dt)
            if len(timing) > 30:
                timing.pop(0)
            fps = 1.0 / (sum(timing)/len(timing))
            cv2.putText(frame, f"FPS:{fps:.1f} | Status: Monitoring", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Wildfire Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # Convert to supervision format and track
        dets = sv.Detections.from_ultralytics(yres[0])
        tracked = tracker.update_with_detections(dets)

        # Process detections
        fire_detected = False
        for xyxy, conf, cls_id, track_id in zip(
            tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id
        ):
            x1, y1, x2, y2 = xyxy.astype(int)

            detection_type = names[cls_id]

            # Check if this is a fire-related detection
            if detection_type.lower() in fire_classes and conf >= args.alert_threshold:
                fire_detected = True
                fire_alert_count += 1
                label = f"FIRE ALERT: {detection_type.upper()} {conf:.2f}"
                # Use red color for fire alerts
                color = (0, 0, 255)  # Red in BGR
            else:
                label = f"{detection_type} {conf:.2f}"
                color = (0, 255, 0)  # Green in BGR

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add status information
        dt = time.time() - t0
        timing.append(dt)
        if len(timing) > 30:
            timing.pop(0)
        fps = 1.0 / (sum(timing)/len(timing))

        status = "FIRE DETECTED!" if fire_detected else "Monitoring"
        status_color = (0, 0, 255) if fire_detected else (0, 255, 0)

        cv2.putText(frame, f"FPS:{fps:.1f} | {status}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.putText(frame, f"Fire Alerts: {fire_alert_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Wildfire Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(
        f"[INFO] Wildfire detection ended. Total fire alerts: {fire_alert_count}")


if __name__ == "__main__":
    main()
