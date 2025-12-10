"""
AprilTag Chef Surprise with OpenCV Hand Gesture Controls
Detects AprilTag 0 to activate Surprise Mode
Uses pure OpenCV for hand gesture recognition (thumbs up/open hand)
No cvzone, no mediapipe - only OpenCV contour analysis
"""

import cv2
import numpy as np
from pyapriltags import Detector
import random
import json
import os
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import time


# ========== CAMERA CONFIGURATION ==========
# IMPORTANT: Change this value to match your camera index
# Common values:
#   0 = Built-in/default camera
#   1 = External USB camera (first one)
#   2 = Second external camera
CAMERA_INDEX = 0  
# ==========================================


# Available ingredients for Chef Surprise
AVAILABLE_INGREDIENTS = {
    "anchovies": "Anchovies",
    "basil": "Fresh Basil",
    "cheese": "Mozzarella Cheese",
    "chicken": "Grilled Chicken",
    "fresh_tomato": "Fresh Tomato",
    "shrimp": "Shrimp"
}


class SimpleHandGestureDetector:
    """
    Pure OpenCV hand gesture detector using contour and convex hull analysis
    Detects thumbs up (1 finger) and open hand (5 fingers) gestures
    """

    def __init__(self):
        # ROI coordinates (x, y, width, height) - LARGER for close camera
        self.roi_x = 350
        self.roi_y = 80
        self.roi_w = 280
        self.roi_h = 280

        # Skin color range in YCrCb color space (more robust than HSV)
        self.lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        # Gesture stability - INCREASED for more reliability
        self.gesture_history = []
        self.stability_frames = 12  # Increased from 8 to 12 for more stable detection

    def detect_hand_roi(self, frame):
        """
        Extract hand from ROI using skin color segmentation
        Returns: (gesture_label, debug_mask, contour)
        """
        # Extract ROI
        roi = frame[self.roi_y:self.roi_y+self.roi_h,
                    self.roi_x:self.roi_x+self.roi_w]

        # Convert to YCrCb color space (better for skin detection)
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

        # Apply skin color mask
        skin_mask = cv2.inRange(ycrcb, self.lower_skin, self.upper_skin)

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Apply Gaussian blur to smooth edges
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, skin_mask, None

        # Get largest contour (assumed to be the hand)
        hand_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(hand_contour)

        # Filter out small contours (noise) - INCREASED threshold
        if area < 5000:
            return None, skin_mask, None

        # Detect gesture from contour
        gesture = self.detect_gesture(hand_contour)

        return gesture, skin_mask, hand_contour

    def detect_gesture(self, contour):
        """
        Classify gesture: THUMBS UP (1 finger) or OPEN (5 fingers)
        Returns: "THUMBSUP", "OPEN", or None
        """
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate aspect ratio to filter out non-hand shapes
            aspect_ratio = w / float(h)

            # More lenient aspect ratio for different hand orientations
            if aspect_ratio > 2.5 or aspect_ratio < 0.35:
                return None

            # Get convex hull with points
            hull = cv2.convexHull(contour, returnPoints=True)

            # Calculate hull area and contour area
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)

            if hull_area == 0:
                return None

            # Solidity = contour_area / hull_area
            # THUMBS UP: moderate solidity
            # OPEN: lower solidity (gaps between fingers)
            solidity = contour_area / hull_area

            # Calculate extent (contour area / bounding box area)
            bbox_area = w * h
            extent = contour_area / bbox_area if bbox_area > 0 else 0

            # Get the center of the hand
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return None

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Count finger tips in upper region
            top_y = y + int(h * 0.65)

            finger_tips = []
            min_distance = 35

            for point in hull:
                px, py = point[0]
                if py < top_y and py < cy:
                    is_new_tip = True
                    for tip in finger_tips:
                        dist = np.sqrt((px - tip[0])**2 + (py - tip[1])**2)
                        if dist < min_distance:
                            is_new_tip = False
                            break

                    if is_new_tip:
                        finger_tips.append((px, py))

            finger_count = len(finger_tips)

            # Calculate perimeter ratio
            perimeter = cv2.arcLength(contour, True)
            hull_perimeter = cv2.arcLength(hull, True)
            perimeter_ratio = perimeter / hull_perimeter if hull_perimeter > 0 else 1

            # ========== THUMBS UP DETECTION (1-2 FINGERS) ==========
            # Thumbs up is much more intuitive and reliable
            # Single finger extended pointing upward
            # Accept 1-2 finger tips (sometimes thumb + small detection artifact)

            # Primary: 1-2 finger tips detected (lenient for thumbs up)
            has_thumb_fingers = finger_count in [1, 2]

            # Secondary: Higher solidity than open hand (closed fist with thumb up)
            # Thumbs up should have moderate to high solidity
            has_good_shape = solidity > 0.72  # Balanced for reliable detection

            # THUMBS UP DETECTION: Reliable detection
            is_thumbs_up = has_thumb_fingers and has_good_shape

            # ========== OPEN HAND DETECTION (SLIGHTLY LENIENT) ==========
            open_criteria = []

            # Critical: Low solidity (gaps between fingers) - slightly more lenient
            if solidity < 0.80:  # Balanced for reliable detection
                open_criteria.append("low_solidity")

            # Critical: 5 fingers visible (or 4 with low solidity)
            if finger_count == 5:
                open_criteria.append("all_five_fingers")
            elif finger_count == 4 and solidity < 0.68:  # Increased from 0.65
                # Accept 4 fingers if solidity is low (indicating strong gaps)
                open_criteria.append("four_fingers_strong_gaps")

            # Supporting: Complex perimeter (jagged from fingers)
            if perimeter_ratio > 1.22:  # Slightly reduced from 1.25
                open_criteria.append("complex_perimeter")

            # Supporting: Aspect ratio closer to square (spread hand)
            if 0.65 <= aspect_ratio <= 1.45:  # Slightly wider range
                open_criteria.append("good_shape")

            # OPEN HAND: Must have low solidity + (5 fingers OR 4 with low solidity) + 1 supporting
            is_strict_open = (
                "low_solidity" in open_criteria and
                ("all_five_fingers" in open_criteria or "four_fingers_strong_gaps" in open_criteria) and
                len(open_criteria) >= 3
            )

            # ========== FINAL CLASSIFICATION ==========
            if is_thumbs_up and not is_strict_open:
                print(f"✅ THUMBS UP 👍 | Sol: {solidity:.2f}, Fingers: {finger_count}")
                return "THUMBSUP"

            elif is_strict_open and not is_thumbs_up:
                print(f"✅ OPEN 🖐 | Sol: {solidity:.2f}, Fingers: {finger_count}")
                return "OPEN"

            else:
                # Concise rejection feedback
                if finger_count in [1, 2] and not has_good_shape:
                    print(f"❌ {finger_count} finger(s) but low solidity ({solidity:.2f}) - close fist more for THUMBS UP")
                elif finger_count == 3:
                    print(f"❌ 3 fingers - spread ALL 5 for OPEN!")
                elif finger_count == 4:
                    print(f"❌ 4 fingers - spread ALL 5 WIDE for OPEN!")
                return None

        except Exception as e:
            print(f"[ERROR] Gesture detection: {e}")
            return None

    def get_stable_gesture(self, current_gesture):
        """
        Return gesture only if stable for required frames
        """
        self.gesture_history.append(current_gesture)

        # Keep only recent history
        if len(self.gesture_history) > self.stability_frames:
            self.gesture_history.pop(0)

        # Check if all recent gestures are the same and not None
        if len(self.gesture_history) == self.stability_frames:
            if current_gesture is not None and all(g == current_gesture for g in self.gesture_history):
                print(f"⭐ STABLE {current_gesture}!")
                return current_gesture

        return None


class ChefSurpriseDetector:
    """
    Main application combining AprilTag detection and hand gesture control
    """

    def __init__(self, camera_id=0):
        print("=" * 70)
        print("CHEF SURPRISE DETECTOR WITH HAND GESTURE CONTROL")
        print("=" * 70)

        # Initialize AprilTag detector
        self.detector = Detector(
            families='tag36h11',
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        self.camera_params = [800, 800, 320, 240]
        self.tag_size = 0.05

        # Initialize hand gesture detector
        self.hand_detector = SimpleHandGestureDetector()

        # Open camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # State management
        self.surprise_mode = False
        self.current_pizza = None
        self.locked_pizza = None
        self.last_gesture = None
        self.gesture_cooldown = 0
        self.cooldown_frames = 30  # Prevent rapid re-triggering
        self.user_wants_checkout = False  # Track if user wants to go to cart
        self.should_exit = False  # Track if we should exit the loop

        # Create orders directory
        self.orders_dir = Path("pizza_orders")
        self.orders_dir.mkdir(exist_ok=True)

        print("[OK] Camera opened successfully")
        print("[OK] Hand gesture detector initialized")
        print("\n📋 INSTRUCTIONS:")
        print("1. Show AprilTag 0 to activate Surprise Mode")
        print("2. Place hand in green ROI box")
        print("3. Show THUMBS UP 👍 to lock in pizza order")
        print("4. Show OPEN HAND 🖐 to generate new pizza")
        print("5. Press 'q' to quit, 'r' to reset")
        print("=" * 70)

    def generate_random_pizza(self):
        """Generate Chef Surprise pizza with exactly 3 random ingredients"""
        num_ingredients = 3
        ingredients = random.sample(list(AVAILABLE_INGREDIENTS.keys()), num_ingredients)

        base_price = 12.99
        price = base_price + (num_ingredients * 2.00)

        return {
            "name": "Chef Surprise",
            "ingredients": ingredients,
            "price": price,
            "timestamp": datetime.now().isoformat()
        }

    def save_order(self, pizza):
        """Save confirmed pizza order to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.orders_dir / f"order_{timestamp}.json"

        order_data = {
            "order_id": timestamp,
            "pizza_name": pizza["name"],
            "ingredients": [AVAILABLE_INGREDIENTS[ing] for ing in pizza["ingredients"]],
            "price": f"${pizza['price']:.2f}",
            "timestamp": pizza["timestamp"]
        }

        with open(filename, 'w') as f:
            json.dump(order_data, f, indent=2)

        print(f"\n💾 [SAVED] Order saved to {filename}")
        return filename

    def detect_tags(self, frame):
        """Detect AprilTags in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size
        )
        return tags

    def draw_roi_box(self, frame):
        """Draw hand placement ROI box"""
        x, y, w, h = (self.hand_detector.roi_x, self.hand_detector.roi_y,
                      self.hand_detector.roi_w, self.hand_detector.roi_h)

        color = (0, 255, 0) if self.surprise_mode else (100, 100, 100)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # ROI label
        label = "PLACE HAND HERE" if self.surprise_mode else "ROI (Inactive)"
        cv2.putText(frame, label, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_pizza_panel(self, frame, pizza):
        """Draw current pizza info panel"""
        panel_x, panel_y, panel_w, panel_h = 10, 10, 380, 180

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x+panel_w, panel_y+panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x+panel_w, panel_y+panel_h), (0, 215, 255), 2)

        # Content
        y = panel_y + 35
        cv2.putText(frame, "CHEF SURPRISE", (panel_x+20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)

        y += 35
        cv2.putText(frame, f"Price: ${pizza['price']:.2f}", (panel_x+20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

        y += 30
        cv2.putText(frame, "Ingredients:", (panel_x+20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

        y += 25
        for i, ing in enumerate(pizza['ingredients']):
            cv2.putText(frame, f"  {i+1}. {AVAILABLE_INGREDIENTS[ing]}",
                       (panel_x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
            y += 25

    def draw_gesture_feedback(self, frame, gesture, stable_gesture):
        """Draw gesture detection feedback"""
        # Position above the controls banner (which is at frame.shape[0] - 50)
        x, y = 10, frame.shape[0] - 90  # 40 pixels above controls banner

        if gesture:
            # Current gesture
            cv2.putText(frame, f"Gesture: {gesture}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Stability indicator
            progress = len(self.hand_detector.gesture_history)
            max_progress = self.hand_detector.stability_frames
            bar_width = int(150 * progress / max_progress)
            cv2.rectangle(frame, (x, y+10), (x+bar_width, y+25), (0, 255, 0), -1)
            cv2.rectangle(frame, (x, y+10), (x+150, y+25), (255, 255, 255), 2)

            if stable_gesture:
                cv2.putText(frame, "STABLE!", (x+160, y+22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def draw_action_feedback(self, frame, action_text, color):
        """Draw large action feedback (LOCKED IN, NEW PIZZA, etc.)"""
        text_size = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 50

        # Background
        cv2.rectangle(frame, (text_x-10, text_y-text_size[1]-10),
                     (text_x+text_size[0]+10, text_y+10), (0, 0, 0), -1)

        # Text
        cv2.putText(frame, action_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)

    def draw_instructions(self, frame):
        """Draw instructions when not in surprise mode"""
        text = "Show AprilTag 0 to start!"
        cv2.putText(frame, text, (150, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def process_hand_gesture(self, gesture, stable_gesture):
        """Process stable hand gestures and trigger actions"""
        if not stable_gesture or self.gesture_cooldown > 0:
            return None

        if stable_gesture == self.last_gesture:
            return None

        if stable_gesture == "THUMBSUP" and self.current_pizza and not self.locked_pizza:
            self.locked_pizza = self.current_pizza.copy()
            self.save_order(self.locked_pizza)
            self.gesture_cooldown = self.cooldown_frames
            self.last_gesture = "THUMBSUP"
            self.hand_detector.gesture_history.clear()
            print(f"\n👍 THUMBS UP CONFIRMED - Order locked in!")

            # Show confirmation dialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            response = messagebox.askyesno(
                "Order Locked!",
                f"Your Chef Surprise pizza is ready!\n\n"
                f"Price: ${self.locked_pizza['price']:.2f}\n"
                f"Ingredients: {', '.join([AVAILABLE_INGREDIENTS[ing] for ing in self.locked_pizza['ingredients']])}\n\n"
                f"Would you like to proceed to checkout?",
                icon='question'
            )
            root.destroy()

            if response:  # User clicked Yes
                self.user_wants_checkout = True
                self.should_exit = True
                print("✅ Proceeding to checkout...")
            else:  # User clicked No - reset
                print("🔄 Resetting order...")
                self.locked_pizza = None
                self.current_pizza = None
                self.surprise_mode = False
                self.hand_detector.gesture_history.clear()

            return "LOCKED IN!"

        elif stable_gesture == "OPEN" and not self.locked_pizza:
            self.current_pizza = self.generate_random_pizza()
            self.gesture_cooldown = self.cooldown_frames
            self.last_gesture = "OPEN"
            self.hand_detector.gesture_history.clear()
            print(f"\n🖐 OPEN HAND CONFIRMED - New pizza!")
            print(f"Price: ${self.current_pizza['price']:.2f}")
            for ing in self.current_pizza['ingredients']:
                print(f" - {AVAILABLE_INGREDIENTS[ing]}")
            return "NEW PIZZA!"

        return None

    def run(self):
        """Main application loop"""
        action_feedback = None
        action_timer = 0

        # FPS tracking
        fps_start_time = time.time()
        fps_frame_count = 0
        fps_display = 0.0

        # Create window with fixed size (cannot be resized/maximized/minimized)
        window_name = "Chef Surprise - Hand Gesture Control"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

        try:
            while True:
                # Check if user wants to exit
                if self.should_exit:
                    break
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Calculate FPS
                fps_frame_count += 1
                fps_elapsed = time.time() - fps_start_time
                if fps_elapsed > 1.0:  # Update FPS every second
                    fps_display = fps_frame_count / fps_elapsed
                    fps_frame_count = 0
                    fps_start_time = time.time()

                # Detect AprilTags BEFORE flipping
                tags = self.detect_tags(frame)

                # Process tag detection
                for tag in tags:
                    if tag.tag_id == 0 and not self.surprise_mode:
                        self.surprise_mode = True
                        self.current_pizza = self.generate_random_pizza()
                        print("\n🎯 SURPRISE MODE ACTIVATED!")
                        print(f"Price: ${self.current_pizza['price']:.2f}")
                        for ing in self.current_pizza['ingredients']:
                            print(f" - {AVAILABLE_INGREDIENTS[ing]}")

                # Flip frame for mirror effect AFTER tag detection
                frame = cv2.flip(frame, 1)

                # Draw ROI box
                self.draw_roi_box(frame)

                # Hand gesture detection
                gesture = None
                stable_gesture = None

                if self.surprise_mode and not self.locked_pizza:
                    gesture, mask, contour = self.hand_detector.detect_hand_roi(frame)

                    if gesture is not None:
                        stable_gesture = self.hand_detector.get_stable_gesture(gesture)
                    else:
                        self.hand_detector.gesture_history.clear()
                        stable_gesture = None

                    if gesture is not None and stable_gesture is not None:
                        new_action = self.process_hand_gesture(gesture, stable_gesture)
                        if new_action:
                            action_feedback = new_action
                            action_timer = 60

                    # Draw hand mask visualization
                    if mask is not None:
                        roi_x, roi_y = self.hand_detector.roi_x, self.hand_detector.roi_y
                        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                        if contour is not None:
                            cv2.drawContours(mask_colored, [contour], -1, (0, 255, 0), 2)
                            hull = cv2.convexHull(contour)
                            cv2.drawContours(mask_colored, [hull], -1, (255, 0, 0), 2)

                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv2.circle(mask_colored, (cx, cy), 5, (0, 0, 255), -1)

                        frame[roi_y:roi_y+self.hand_detector.roi_h,
                              roi_x:roi_x+self.hand_detector.roi_w] = \
                            cv2.addWeighted(frame[roi_y:roi_y+self.hand_detector.roi_h,
                                                  roi_x:roi_x+self.hand_detector.roi_w],
                                          0.5, mask_colored, 0.5, 0)

                    self.draw_gesture_feedback(frame, gesture, stable_gesture)

                # Draw pizza info
                if self.current_pizza and not self.locked_pizza:
                    self.draw_pizza_panel(frame, self.current_pizza)
                elif self.locked_pizza:
                    self.draw_pizza_panel(frame, self.locked_pizza)
                    cv2.putText(frame, "ORDER CONFIRMED!", (120, 240),
                               cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)

                if not self.surprise_mode:
                    self.draw_instructions(frame)

                # Draw action feedback
                if action_feedback and action_timer > 0:
                    color = (0, 255, 0) if "LOCKED" in action_feedback else (0, 215, 255)
                    self.draw_action_feedback(frame, action_feedback, color)
                    action_timer -= 1
                    if action_timer == 0:
                        action_feedback = None

                # Update cooldown
                if self.gesture_cooldown > 0:
                    self.gesture_cooldown -= 1
                else:
                    self.last_gesture = None

                # Draw FPS counter in top-right corner
                fps_text = f"FPS: {fps_display:.1f}"
                fps_text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                fps_x = frame.shape[1] - fps_text_size[0] - 10
                fps_y = 30
                # Background for FPS
                cv2.rectangle(frame, (fps_x - 5, fps_y - 20), (fps_x + fps_text_size[0] + 5, fps_y + 5), (40, 40, 40), -1)
                cv2.putText(frame, fps_text, (fps_x, fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw keyboard shortcuts at the bottom
                shortcuts_y_start = frame.shape[0] - 50
                cv2.rectangle(frame, (0, shortcuts_y_start), (frame.shape[1], frame.shape[0]), (40, 40, 40), -1)
                cv2.putText(frame, "Controls: [Q] Quit  |  [R] Reset", (10, shortcuts_y_start + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.surprise_mode = False
                    self.current_pizza = None
                    self.locked_pizza = None
                    self.hand_detector.gesture_history.clear()
                    print("\n🔄 [RESET] System reset")

        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n👋 [EXIT] Application closed")


def main():
    try:
        detector = ChefSurpriseDetector(camera_id=CAMERA_INDEX)
        detector.run()
    except Exception as e:
        print(f"\n❌ [ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
