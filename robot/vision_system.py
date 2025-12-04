"""
ChefMate Robot Assistant - Vision System
Handles camera operations and YOLO-based ingredient detection
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple
import time

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Warning: Ultralytics not found. Detection will be simulated.")

from config.settings import (
    CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IMAGE_SIZE, INGREDIENT_CLASSES
)


class VisionSystem:
    """
    Vision system for ingredient detection using YOLOv11n
    Handles camera operations and real-time object detection
    """

    def __init__(self):
        """Initialize the vision system"""
        self.logger = logging.getLogger(__name__)
        self.camera = None
        self.model = None
        self.is_camera_active = False
        self.is_model_loaded = False
        self.class_names = INGREDIENT_CLASSES

        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

        # Load YOLO model
        self.load_model()

    def load_model(self) -> bool:
        """
        Load YOLOv11n model

        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not YOLO_AVAILABLE:
                self.logger.warning("YOLO not available - running in simulation mode")
                self.is_model_loaded = False
                return False

            self.logger.info(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
            self.model = YOLO(YOLO_MODEL_PATH)

            # Warmup inference
            dummy_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False, imgsz=YOLO_IMAGE_SIZE)

            self.is_model_loaded = True
            self.logger.info("✅ YOLO model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.is_model_loaded = False
            return False

    def start_camera(self, camera_id: int = CAMERA_ID) -> bool:
        """
        Start the camera with optimized settings

        Args:
            camera_id: Camera device ID (default: 0)

        Returns:
            bool: True if camera started successfully
        """
        try:
            self.logger.info(f"Starting camera {camera_id}...")

            # Try default backend first (more reliable)
            self.camera = cv2.VideoCapture(camera_id)

            if not self.camera.isOpened():
                self.logger.error(f"Failed to open camera {camera_id}")
                return False

            # Configure camera - minimal settings for compatibility
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimal buffer for low latency

            # Test read
            ret, frame = self.camera.read()
            if not ret or frame is None:
                self.logger.error("Camera opened but cannot read frames")
                self.camera.release()
                return False

            self.is_camera_active = True
            self.logger.info(f"✅ Camera started: {frame.shape[1]}x{frame.shape[0]}")
            return True

        except Exception as e:
            self.logger.error(f"Error starting camera: {e}")
            return False

    def stop_camera(self):
        """Stop the camera"""
        if self.camera:
            self.logger.info("Stopping camera...")
            self.camera.release()
            self.camera = None
            self.is_camera_active = False
            self.logger.info("Camera stopped")

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from the camera

        Returns:
            Frame as numpy array or None if failed
        """
        if not self.is_camera_active or not self.camera:
            return None

        try:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time

                return frame
            return None
        except Exception as e:
            self.logger.error(f"Error reading frame: {e}")
            return None

    def detect_ingredient(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Detect ingredient in frame using YOLO

        Args:
            frame: Input image frame

        Returns:
            Tuple of (annotated_frame, detection_dict)
            detection_dict format: {
                'class_name': str,
                'confidence': float,
                'bbox': [x1, y1, x2, y2]
            }
        """
        if not self.is_model_loaded or self.model is None:
            # Simulation mode - draw fake detection
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "SIMULATION MODE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_frame, None

        try:
            # Run YOLO inference
            results = self.model.predict(
                frame,
                conf=YOLO_CONFIDENCE_THRESHOLD,
                imgsz=YOLO_IMAGE_SIZE,
                verbose=False,
                max_det=1,  # Only detect top ingredient
                agnostic_nms=True
            )

            annotated_frame = frame.copy()
            detection = None

            if results and len(results) > 0:
                result = results[0]

                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    # Get highest confidence detection
                    box = result.boxes[0]
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get class name
                    if hasattr(result, 'names'):
                        class_name = result.names[class_id]
                    else:
                        class_name = self.class_names[class_id] if class_id < len(
                            self.class_names) else f"Class_{class_id}"

                    # Store detection info
                    detection = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id
                    }

                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label with background
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_frame,
                                  (x1, y1 - label_size[1] - 10),
                                  (x1 + label_size[0], y1),
                                  color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return annotated_frame, detection

        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
            return frame, None

    def capture_and_detect(self) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Capture a frame and run detection

        Returns:
            Tuple of (annotated_frame, detection_dict)
        """
        frame = self.read_frame()
        if frame is None:
            return None, None

        return self.detect_ingredient(frame)

    def detect_current_ingredient(self, confidence_threshold: float = None) -> Optional[str]:
        """
        Detect ingredient in current camera view (single attempt)

        Args:
            confidence_threshold: Minimum confidence (uses default if None)

        Returns:
            Detected ingredient class name or None if no detection
        """
        if confidence_threshold is None:
            confidence_threshold = YOLO_CONFIDENCE_THRESHOLD

        frame, detection = self.capture_and_detect()

        if detection is None:
            return None

        detected_class = detection['class_name']
        confidence = detection['confidence']

        if confidence >= confidence_threshold:
            self.logger.info(f"Detected: {detected_class} ({confidence:.2f})")
            return detected_class

        return None

    def verify_ingredient(self, expected_ingredient: str, max_attempts: int = 3) -> bool:
        """
        Verify that the expected ingredient is detected

        Args:
            expected_ingredient: Name of expected ingredient
            max_attempts: Maximum number of detection attempts

        Returns:
            bool: True if correct ingredient detected
        """
        self.logger.info(f"Verifying ingredient: {expected_ingredient}")

        for attempt in range(max_attempts):
            frame, detection = self.capture_and_detect()

            if detection is None:
                self.logger.warning(f"Attempt {attempt + 1}/{max_attempts}: No detection")
                time.sleep(0.2)
                continue

            detected_class = detection['class_name']
            confidence = detection['confidence']

            self.logger.info(f"Detected: {detected_class} ({confidence:.2f})")

            if detected_class == expected_ingredient:
                self.logger.info(f"✅ Correct ingredient verified: {expected_ingredient}")
                return True
            else:
                self.logger.warning(f"Wrong ingredient! Expected {expected_ingredient}, got {detected_class}")

        self.logger.error(f"Failed to verify {expected_ingredient} after {max_attempts} attempts")
        return False

    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps

    def is_ready(self) -> bool:
        """Check if vision system is ready"""
        return self.is_camera_active and self.is_model_loaded

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_camera()