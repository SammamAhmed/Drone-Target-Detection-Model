#!/usr/bin/env python3
"""
Batch test fire detection on wildfire dataset images
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def create_fire_mask(hsv_frame):
    """Create fire detection mask"""
    fire_ranges = [
        ([5, 100, 100], [25, 255, 255]),    # Orange
        ([0, 120, 70], [10, 255, 255]),     # Red
        ([160, 120, 70], [180, 255, 255]),  # Red (wraparound)
        ([20, 100, 100], [30, 255, 255]),   # Yellow-Orange
    ]

    combined_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

    for lower, upper in fire_ranges:
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv_frame, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    return combined_mask


def test_image(img_path, show_results=False):
    """Test fire detection on single image"""
    img = cv2.imread(str(img_path))
    if img is None:
        return 0, False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    fire_mask = create_fire_mask(hsv)

    # Calculate fire percentage
    fire_pixels = cv2.countNonZero(fire_mask)
    total_pixels = img.shape[0] * img.shape[1]
    fire_percentage = (fire_pixels / total_pixels) * 100

    # Find fire regions
    contours, _ = cv2.findContours(
        fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_regions = [c for c in contours if cv2.contourArea(c) > 500]

    fire_detected = fire_percentage > 2.0 and len(large_regions) > 0

    if show_results:
        # Draw results on image
        result_img = img.copy()

        for contour in large_regions:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result_img, f"FIRE {fire_percentage:.1f}%",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show fire mask overlay
        fire_overlay = cv2.applyColorMap(fire_mask, cv2.COLORMAP_HOT)
        combined = cv2.addWeighted(result_img, 0.7, fire_overlay, 0.3, 0)

        cv2.imshow(f'Fire Detection: {img_path.name}', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return fire_percentage, fire_detected


def main():
    parser = argparse.ArgumentParser(description='Batch test fire detection')
    parser.add_argument('--dataset-path', default='Dataset',
                        help='Dataset root path')
    parser.add_argument('--show', action='store_true',
                        help='Show detection results')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to test')

    args = parser.parse_args()

    dataset_root = Path(args.dataset_path)

    print("Fire Detection Batch Test")
    print("=" * 40)

    # Test wildfire images
    wildfire_path = dataset_root / "test" / "wildfire"
    if wildfire_path.exists():
        print(f"\nTesting WILDFIRE images from: {wildfire_path}")
        wildfire_images = list(wildfire_path.glob("*.jpg"))[:args.num_samples]

        correct_detections = 0
        total_fire_percent = 0

        for i, img_path in enumerate(wildfire_images, 1):
            fire_percent, detected = test_image(img_path, args.show)
            total_fire_percent += fire_percent

            status = "DETECTED" if detected else "MISSED"
            print(
                f"{i:2d}. {img_path.name[:30]:30} - {fire_percent:5.1f}% - {status}")

            if detected:
                correct_detections += 1

        if wildfire_images:
            accuracy = (correct_detections / len(wildfire_images)) * 100
            avg_fire_percent = total_fire_percent / len(wildfire_images)
            print(
                f"\nWildfire Detection Accuracy: {correct_detections}/{len(wildfire_images)} ({accuracy:.1f}%)")
            print(f"Average Fire Percentage: {avg_fire_percent:.1f}%")

    # Test no-wildfire images
    nowildfire_path = dataset_root / "test" / "nowildfire"
    if nowildfire_path.exists():
        print(f"\nTesting NO-WILDFIRE images from: {nowildfire_path}")
        nowildfire_images = list(
            nowildfire_path.glob("*.jpg"))[:args.num_samples]

        correct_rejections = 0
        total_false_fire_percent = 0

        for i, img_path in enumerate(nowildfire_images, 1):
            fire_percent, detected = test_image(img_path, args.show)
            total_false_fire_percent += fire_percent

            status = "FALSE POSITIVE" if detected else "CORRECT"
            print(
                f"{i:2d}. {img_path.name[:30]:30} - {fire_percent:5.1f}% - {status}")

            if not detected:
                correct_rejections += 1

        if nowildfire_images:
            accuracy = (correct_rejections / len(nowildfire_images)) * 100
            avg_false_fire_percent = total_false_fire_percent / \
                len(nowildfire_images)
            print(
                f"\nNo-Fire Detection Accuracy: {correct_rejections}/{len(nowildfire_images)} ({accuracy:.1f}%)")
            print(
                f"Average False Fire Percentage: {avg_false_fire_percent:.1f}%")

    print("\n" + "=" * 40)
    print("This color-based method provides basic fire detection")
    print("Adjust thresholds in the script for better accuracy")
    print("Use --show flag to see visual results")


if __name__ == "__main__":
    main()
