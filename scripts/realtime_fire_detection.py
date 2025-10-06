#!/usr/bin/env python3
"""
Real-time Fire Detection for Webcam
This script provides working fire detection for visible flames using color analysis
"""

import cv2
import numpy as np
import argparse
import time


def create_fire_mask(hsv_frame):
    """Create enhanced fire detection mask for real flames"""
    # More aggressive fire color ranges for real flames
    fire_ranges = [
        # Bright orange (main fire color)
        ([8, 150, 150], [20, 255, 255]),
        # Red-orange
        ([0, 150, 150], [8, 255, 255]),
        # Red (wraparound)
        ([170, 150, 150], [180, 255, 255]),
        # Yellow flames
        ([20, 100, 200], [30, 255, 255]),
        # Bright red
        ([160, 150, 150], [170, 255, 255]),
    ]

    combined_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

    for lower, upper in fire_ranges:
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv_frame, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Enhanced morphological operations for flame shapes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, kernel_close)
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, kernel_open)

    return combined_mask


def detect_motion_in_fire_regions(prev_gray, curr_gray, fire_mask):
    """
    Detect motion in fire-colored regions to improve accuracy
    Real flames flicker/move, static orange objects don't
    """
    if prev_gray is None:
        return 0

    # Calculate optical flow in fire regions
    flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray,
                                    corners=None, winSize=(15, 15),
                                    maxLevel=2,
                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Simple motion detection using frame difference
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Combine fire mask with motion
    fire_motion = cv2.bitwise_and(fire_mask, motion_mask)
    motion_pixels = cv2.countNonZero(fire_motion)

    return motion_pixels


def main():
    parser = argparse.ArgumentParser(description='Real-time Fire Detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--fire-threshold', type=float,
                        default=1.0, help='Fire detection threshold (%)')
    parser.add_argument('--motion-threshold', type=int,
                        default=100, help='Motion pixels threshold')
    parser.add_argument('--sensitivity', type=str, default='medium',
                        choices=['low', 'medium', 'high'], help='Detection sensitivity')

    args = parser.parse_args()

    # Adjust thresholds based on sensitivity
    if args.sensitivity == 'low':
        args.fire_threshold = 2.0
        args.motion_threshold = 200
    elif args.sensitivity == 'high':
        args.fire_threshold = 0.5
        args.motion_threshold = 50

    print(f" Real-time Fire Detection Starting...")
    print(f" Camera: {args.camera}")
    print(f" Fire threshold: {args.fire_threshold}%")
    print(f" Motion threshold: {args.motion_threshold} pixels")
    print(f" Sensitivity: {args.sensitivity}")
    print(f" Press 'q' to quit, 'c' to calibrate, 's' for screenshot")

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f" Error: Could not open camera {args.camera}")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_gray = None
    frame_count = 0
    fire_alert_count = 0
    last_alert_time = 0

    print(" Camera opened successfully. Starting detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to read from camera")
            break

        frame_count += 1
        current_time = time.time()

        # Convert to HSV and grayscale
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create fire mask
        fire_mask = create_fire_mask(hsv)

        # Calculate fire percentage
        fire_pixels = cv2.countNonZero(fire_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        fire_percentage = (fire_pixels / total_pixels) * 100

        # Detect motion in fire regions
        motion_pixels = detect_motion_in_fire_regions(
            prev_gray, gray, fire_mask)

        # Find fire contours
        contours, _ = cv2.findContours(
            fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_fire_regions = [c for c in contours if cv2.contourArea(c) > 200]

        # Determine if fire is detected
        fire_detected = (fire_percentage > args.fire_threshold and
                         len(large_fire_regions) > 0 and
                         motion_pixels > args.motion_threshold)

        # Draw fire regions
        for contour in large_fire_regions:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Color based on confidence
            if fire_detected:
                color = (0, 0, 255)  # Red for confirmed fire
                label = f"FIRE! {fire_percentage:.1f}%"
            else:
                color = (0, 165, 255)  # Orange for possible fire
                label = f"Possible {fire_percentage:.1f}%"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw status and info
        if fire_detected:
            fire_alert_count += 1
            if current_time - last_alert_time > 2:  # Alert every 2 seconds
                print(
                    f" FIRE ALERT! Frame {frame_count} - {fire_percentage:.2f}% fire, {motion_pixels} motion pixels")
                last_alert_time = current_time

            status = f" FIRE DETECTED! ({fire_percentage:.1f}%)"
            status_color = (0, 0, 255)
        else:
            status = f" Monitoring ({fire_percentage:.1f}%)"
            status_color = (0, 255, 0)

        # Draw status text
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # Draw detection bars
        bar_y = 50
        # Fire percentage bar
        cv2.putText(frame, f"Fire: {fire_percentage:.1f}%", (10, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        fire_bar_width = int(fire_percentage * 3)  # Scale for display
        cv2.rectangle(frame, (80, bar_y),
                      (80 + fire_bar_width, bar_y + 10), (0, 0, 255), -1)

        # Motion bar
        motion_bar_y = bar_y + 20
        cv2.putText(frame, f"Motion: {motion_pixels}", (10, motion_bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        motion_bar_width = min(200, motion_pixels // 2)
        cv2.rectangle(frame, (80, motion_bar_y), (80 +
                      motion_bar_width, motion_bar_y + 10), (0, 255, 255), -1)

        # Show fire mask overlay
        fire_overlay = cv2.applyColorMap(fire_mask, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(frame, 0.8, fire_overlay, 0.2, 0)

        cv2.imshow('Fire Detection', overlay)

        # Update previous frame
        prev_gray = gray.copy()

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print(" Calibrating... Look around the room to set baseline")
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f'fire_detection_{timestamp}.jpg'
            cv2.imwrite(filename, frame)
            print(f" Screenshot saved: {filename}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n Detection Summary:")
    print(f"   Total frames: {frame_count}")
    print(f"   Fire alerts: {fire_alert_count}")
    print(f"   Alert rate: {(fire_alert_count/frame_count*100):.1f}%")
    print(" Fire detection stopped")


if __name__ == "__main__":
    main()
