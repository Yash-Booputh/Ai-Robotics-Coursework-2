# test_pick_sequence_audio.py
import logging
import time
from robot_music import PizzaRobotAudio
# Mock the other imports for testing

logging.basicConfig(level=logging.INFO)

# Test audio with pick sequence simulation
print("Testing PickSequence audio integration...")

# Initialize audio
audio = PizzaRobotAudio()



print("\n2. Finding ingredient...")
audio.play_found_cube()
time.sleep(6)

print("\n3. Delivering ingredient...")
audio.play_drop_cube()
time.sleep(6)

print("\n4. Order complete...")
audio.play_pizza_sound("Chicken Supreme")

print("\n✅ Pick sequence audio test complete!")
audio.cleanup()