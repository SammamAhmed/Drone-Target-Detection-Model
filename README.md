# Drone-Target-Detection-Model
This fire detection system is not fully trained or complete yet. The custom YOLO models are still being developed and the training dataset needs improvements.

## What Currently Works

### Real-Time Fire Detection
- **Script:** `scripts/realtime_fire_detection.py`
- **Method:** Color-based detection using HSV analysis
- **Status:** Functional for visible flames
- **Usage:** `python scripts/realtime_fire_detection.py --camera 0`

### Basic Color Detection
- **Script:** `scripts/color_fire_detection.py`
- **Method:** HSV color space analysis
- **Status:** Working for orange/red flames
