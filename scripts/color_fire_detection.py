#!/usr/bin/env python3
"""
Color-based Fire Detection Script
This script uses color analysis to detect fire/flames in real-time video or images
It works best for visible flames (orange/red) and may not detect smoke or distant fires
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import time


def create_fire_mask(hsv_frame):
    """
    Create a mask for fire-colored pixels using HSV color space
    Returns binary mask where white pixels indicate potential fire
    """
    # Fire color ranges in HSV
    fire_ranges = [
        # Orange/Red-Orange (main fire colors)
        ([5, 100, 100], [25, 255, 255]),    # Orange
        ([0, 120, 70], [10, 255, 255]),     # Red
        ([160, 120, 70], [180, 255, 255]),  # Red (wraparound)
        ([20, 100, 100], [30, 255, 255]),   # Yellow-Orange
    ]

    # Combine all fire color masks
    combined_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

    for lower, upper in fire_ranges:
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv_frame, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    return combined_mask


def detect_fire_regions(fire_mask, min_area=500):
    """
    Find fire regions in the binary mask
    Returns list of contours representing fire regions
    """
    # Find contours
    contours, _ = cv2.findContours(
        fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    fire_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            fire_regions.append(contour)

    return fire_regions


def calculate_fire_percentage(fire_mask, frame_shape):
    """Calculate percentage of frame that contains fire colors"""
    fire_pixels = cv2.countNonZero(fire_mask)
    total_pixels = frame_shape[0] * frame_shape[1]
    return (fire_pixels / total_pixels) * 100


def draw_fire_detections(frame, fire_regions, fire_percentage):
    """Draw bounding boxes and labels for detected fire regions"""
    fire_detected = len(fire_regions) > 0

    # Draw fire regions
    for contour in fire_regions:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Calculate confidence based on area and fire percentage
        area = cv2.contourArea(contour)
        confidence = min(0.99, (area / 10000) * (fire_percentage / 10))

        # Draw label
        label = f"FIRE {confidence:.2f}"
        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - label_size[1] - 10),
                      (x + label_size[0], y), (0, 0, 255), -1)
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw status
    if fire_detected:
        status = f"FIRE DETECTED! ({fire_percentage:.1f}%)"
        color = (0, 0, 255)  # Red
    else:
        status = f"Monitoring ({fire_percentage:.1f}%)"
        color = (0, 255, 0)  # Green

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Draw fire percentage bar
    bar_width = 200
    bar_height = 20
    bar_x, bar_y = 10, 50

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width,
                  bar_y + bar_height), (100, 100, 100), -1)

    # Fill based on fire percentage
    fill_width = int((fire_percentage / 50.0) * bar_width)  # 50% = full bar
    fill_color = (0, 255 - int(fire_percentage * 5), int(fire_percentage * 5))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width,
                  bar_y + bar_height), fill_color, -1)

    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width,
                  bar_y + bar_height), (255, 255, 255), 2)

    return fire_detected


def main():
    parser = argparse.ArgumentParser(description='Color-based Fire Detection')
    parser.add_argument('--source', default='0',
                        help='Video source (0 for webcam, path for video file)')
    parser.add_argument('--min-area', type=int, default=500,
                        help='Minimum fire region area')
    parser.add_argument('--fire-threshold', type=float,
                        default=2.0, help='Fire detection threshold (%)')
    parser.add_argument('--save', help='Save output video to file')

    args = parser.parse_args()

    # Open video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return

    print("Color-based Fire Detection Started")
    print("Press 'q' to quit, 's' to save screenshot")
    print(f"Fire threshold: {args.fire_threshold}%")
    print(f"Minimum area: {args.min_area} pixels")

    # Video writer setup
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create fire mask
        fire_mask = create_fire_mask(hsv)

        # Calculate fire percentage
        fire_percentage = calculate_fire_percentage(fire_mask, frame.shape)

        # Detect fire regions
        fire_regions = detect_fire_regions(fire_mask, args.min_area)

        # Draw detections
        fire_detected = draw_fire_detections(
            frame, fire_regions, fire_percentage)

        # Alert if fire detected
        if fire_detected and fire_percentage > args.fire_threshold:
            print(
                f"Frame {frame_count}: FIRE DETECTED! ({fire_percentage:.2f}%)")

        # Show FPS
        if frame_count % 30 == 0:  # Update every 30 frames
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f}, Fire regions: {len(fire_regions)}")

        # Display frame
        cv2.imshow('Fire Detection', frame)

        # Save video frame
        if writer:
            writer.write(frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f'fire_detection_{timestamp}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames")
    print("Fire detection stopped")


if __name__ == "__main__":
    main()
