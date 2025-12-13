# ChefMate Robot Pizza Maker

> **"One Slice and—Mamma Mia!"**

An intelligent robotic pizza maker and preparation system powered by YOLOv11n object detection and the Yahboom Dofbot robotic arm.

## Video Demonstration
Watch the full system demonstration and presentation:
[ChefMate Robot Assistant - YouTube](https://youtu.be/2Bl7t31zxcY)

## Repository
GitHub Repository:
[Yash-Booputh/Ai-Robotics-Coursework-2](https://github.com/Yash-Booputh/Ai-Robotics-Coursework-2)

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Hardware Setup](#hardware-setup)
- [Pizza Menu](#pizza-menu)
- [Troubleshooting](#troubleshooting)

---

## Overview

ChefMate is a comprehensive pizza ordering and robotic preparation system that combines:
- Modern GUI with multiple ordering methods
- YOLOv11 object detection for ingredient recognition
- Yahboom Dofbot robotic arm for autonomous ingredient picking
- AprilTag detection for visual pizza ordering
- Audio feedback system with background music and notifications

### Course Information
- **Course**: AI in Robotics (PDE 3802)
- **Institution**: Middlesex University
- **Submission Date**: December 12, 2025

---

## Features

### User Interface
- Loading screen with animated progress bar
- Home screen with multiple operation modes
- Color-coded interface with intuitive navigation
- Integrated volume control in title bar
- Real-time status updates and progress tracking

### Ordering Methods

1. **Traditional Menu Ordering**
   - Browse specialty pizzas with descriptions
   - Add items to cart and review order
   - Watch robot execute order in real-time

2. **AprilTag Visual Ordering**
   - Point camera at AprilTag markers
   - Each tag corresponds to a specific pizza
   - Instant visual confirmation

3. **File Upload Detection**
   - Upload ingredient images for testing
   - View detection results with confidence scores

4. **Live Camera View**
   - Real-time ingredient detection
   - Live FPS and confidence display

### Robot Capabilities
- Autonomous scanning of 6 shelf positions for ingredients
- Sequential picking and collection
- Delivery to designated area
- Collision avoidance and error handling

### Audio System
- Background music
- Pizza-specific notification sounds
- Status update audio cues
- Adjustable volume control (0-100%)

---

## Installation

### System Requirements
- **Operating System**: Raspberry Pi OS (64-bit) or Windows/Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (Raspberry Pi 4 recommended)
- **Camera**: USB camera or Raspberry Pi Camera Module

### Step 1: Clone Repository
```bash
cd ~/Desktop
git clone https://github.com/Yash-Booputh/Ai-Robotics-Coursework-2.git
cd Ai-Robotics-Coursework-2
```

### Step 2: Set Up Virtual Environment
It is recommended to run the application in a virtual environment to avoid dependency conflicts.

**On Linux/Raspberry Pi:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

**On Windows:**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate
```

### Step 3: Install Dependencies
With the virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `ultralytics` - YOLOv11 framework
- `opencv-python` - Computer vision and camera handling
- `pillow` - Image processing for UI
- `pygame` - Audio playback system
- `pupil-apriltags` - AprilTag detection
- `numpy` - Numerical operations
- `Arm_Lib` - Yahboom Dofbot control (robot hardware only)

### Step 4: Add Required Assets

Place your trained YOLOv11 model:
```bash
models/best.pt
```

Add application logo:
```bash
assets/logo.png
```

Add audio files (optional but recommended):
```bash
robot_sounds/background_music.mp3
robot_sounds/greetings.wav
robot_sounds/margherita.wav
robot_sounds/chicken_supreme.wav
robot_sounds/seafood.wav
robot_sounds/anchovy.wav
robot_sounds/pesto.wav
robot_sounds/ocean_garden.wav
robot_sounds/found.wav
robot_sounds/delivery_done.wav
robot_sounds/error_1.wav
```

---

## Usage

### Running the Application

**IMPORTANT:** Always run the application from within the virtual environment.

**On Linux/Raspberry Pi:**
```bash
# Activate virtual environment (if not already activated)
source .venv/bin/activate

# Run the application
.venv/bin/python main.py
```

**On Windows:**
```bash
# Activate virtual environment (if not already activated)
.venv\Scripts\activate

# Run the application
.venv\Scripts\python main.py
```

**Alternative method (with activated environment):**
```bash
# After activating the virtual environment
python main.py
```

### Application Flow

1. **Loading Screen**
   - Displays ChefMate logo and slogan
   - Animated progress bar
   - Application initialization

2. **Home Screen**
   - Select from available operation modes:
     - Order Pizza - Traditional menu ordering
     - Live Camera - Real-time detection view
     - File Upload - Test detection on images

3. **Order Execution**
   - Robot screen shows live camera feed
   - Progress checklist tracks each ingredient
   - Audio notifications for each step
   - Success/failure status updates

### Deactivating Virtual Environment
When finished, deactivate the virtual environment:
```bash
deactivate
```

---

## Project Structure

```
Ai-Robotics-Coursework-2/
│
├── main.py                                    # Application entry point
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
│
├── .venv/                                      # Virtual environment (created after setup)
│
├── assets/
│   └── logo.png                                # ChefMate logo
│
├── config/
│   ├── __init__.py
│   ├── settings.py                             # Application settings
│   └── recipes.py                              # Pizza recipes and ingredients
│
├── robot/
│   ├── __init__.py
│   └── vision_system.py                        # YOLOv11 detection system
│
├── ui/
│   ├── __init__.py
│   ├── loading_screen.py                       # Startup loading screen
│   ├── home_screen.py                          # Main menu
│   ├── menu_screen.py                          # Pizza menu
│   ├── cart_screen.py                          # Order review
│   ├── robot_screen.py                         # Robot execution view
│   ├── file_upload_screen.py                   # Image upload for testing
│   ├── live_camera_screen.py                   # Live detection view
│   └── widgets.py                              # Reusable UI components
│
├── models/
│   └── best.pt                                 # YOLOv11 trained model
│
├── robot_sounds/                               # Audio files (optional)
│   ├── background_music.mp3
│   ├── greetings.wav
│   ├── margherita.wav
│   ├── chicken_supreme.wav
│   ├── seafood.wav
│   ├── anchovy.wav
│   ├── pesto.wav
│   ├── ocean_garden.wav
│   ├── found.wav
│   ├── delivery_done.wav
│   └── error_1.wav
│
├── pizza_orders/                               # Saved orders (generated at runtime)
│
├── logs/                                       # Application logs (generated at runtime)
│   └── chefmate.log
│
├── apriltag_pizza_detector_simple.py           # AprilTag detection script
├── robot_music.py                              # Audio system controller
├── integrated_patrol_grab.py                   # Robot patrol and grab system
├── grab_position_configurator_controller.py    # Grab position calibration tool
├── slot_position_configurator_controller.py    # Slot position calibration tool
├── grab_positions.json                         # Saved grab waypoints
└── slot_positions.json                         # Saved slot positions
```

---

## Configuration

### Application Settings
Edit [config/settings.py](config/settings.py) to customize:

```python
# Window Settings
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768

# Camera Settings
CAMERA_ID = 2                         # Camera device ID (adjust as needed)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# YOLO Detection
YOLO_MODEL_PATH = "models/best.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.5       # Minimum confidence (0.0-1.0)
YOLO_IMAGE_SIZE = 320                 # 224, 320, or 640
```

### Pizza Recipes
Edit [config/recipes.py](config/recipes.py) to add or modify pizzas:

```python
PIZZA_RECIPES = {
    "Margherita": {
        "name": "Margherita",
        "description": "Classic Italian pizza with tomato, cheese and fresh basil",
        "ingredients": ["fresh_tomato", "cheese", "basil"],
        "price": "$12.99"
    }
    # Add more pizzas as needed
}
```

### Robot Position Calibration
Before using the robot, calibrate slot and grab positions:

```bash
# Calibrate slot scanning positions
.venv/bin/python slot_position_configurator_controller.py

# Calibrate grab waypoints for each slot
.venv/bin/python grab_position_configurator_controller.py
```

Positions are saved to:
- `slot_positions.json` - Scanning positions for each shelf slot
- `grab_positions.json` - 3-waypoint grab sequences for each slot

---

## Hardware Setup

### Shelf Layout
The system uses a 6-slot shelf arranged in 3 rows:

```
┌─────────────────────────────────┐
│  [Slot 1]        [Slot 2]       │  Top Row
│  [Slot 3]        [Slot 4]       │  Middle Row
│  [Slot 5]        [Slot 6]       │  Bottom Row
│         [Delivery Area]         │
└─────────────────────────────────┘
```

### Camera Setup
- Mount camera on robot arm end effector
- Camera should face downward for ingredient detection
- Ensure stable mounting to prevent blur
- Test camera feed before calibration

### Ingredient Cubes
- 6 ingredient types on physical cubes
- Each cube displays ingredient image on top face
- Place cubes randomly in shelf slots
- Ensure adequate lighting for detection

### Important Safety Notes
- Keep emergency stop button accessible
- Never run with uncalibrated positions
- Monitor robot during initial runs
- Keep workspace clear of obstacles

---

## Pizza Menu

| Pizza | Ingredients | Price |
|-------|-------------|-------|
| Margherita | Fresh Tomato, Cheese, Basil | $12.99 |
| Chicken Supreme | Fresh Tomato, Cheese, Chicken | $14.99 |
| Seafood Delight | Fresh Tomato, Cheese, Shrimp | $16.99 |
| Anchovy Special | Fresh Tomato, Cheese, Anchovies | $13.99 |
| Pesto Chicken | Chicken, Cheese, Basil | $15.99 |
| Ocean Garden | Shrimp, Anchovies, Basil | $17.99 |

### Available Ingredients
- Anchovies
- Basil
- Cheese
- Chicken
- Fresh Tomato
- Shrimp

---

## Troubleshooting

### Common Issues

#### Camera Not Detected
Check available cameras on your system:

**Linux/Raspberry Pi:**
```bash
ls /dev/video*
```

**Test camera:**
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

**Solution:** Update `CAMERA_ID` in [config/settings.py](config/settings.py:46)

#### Low Detection Performance
- Reduce `YOLO_IMAGE_SIZE` to 224 in settings
- Close unnecessary applications
- Reduce camera resolution
- Check system temperature

#### False Detections
- Increase `YOLO_CONFIDENCE_THRESHOLD` in settings
- Improve lighting conditions
- Ensure ingredient images are clear and visible

#### Module Import Errors
Reinstall dependencies in the virtual environment:
```bash
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt --force-reinstall
```

#### Robot Not Responding
Check robot hardware connection:
```bash
# Activate virtual environment first
source .venv/bin/activate

# Test Arm_Lib import
python -c "from Arm_Lib import Arm_Device; print('OK')"

# Check USB connection (Linux)
ls /dev/ttyUSB*
```

#### Inaccurate Robot Movements
- Recalibrate all positions using the configuration tools
- Check servo power supply
- Verify all servo connections
- Reduce movement speed in code if needed

#### Audio Not Playing
Verify audio system:
```bash
python -c "import pygame; pygame.mixer.init(); print('OK')"
```

Check that audio files exist in `robot_sounds/` directory

### Application Logs
Check logs for detailed error information:
```bash
tail -f logs/chefmate.log
```

---

## Technologies Used
- Python 3.8+
- YOLOv11 (Ultralytics)
- OpenCV
- Tkinter (GUI)
- Pygame (Audio)
- AprilTag Detection
- Yahboom Dofbot

---

## Course Information
- **Course**: AI in Robotics (PDE 3802)
- **Institution**: Middlesex University
- **Submission Date**: December 12, 2025

---

