import pygame
import time
import os
from pathlib import Path

def test_laptop_speaker():
    """Simple test to verify audio works on your laptop"""
    
    print("🔊 Laptop Speaker Test")
    print("=" * 40)
    
    # Initialize pygame mixer
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        print("✓ Pygame mixer initialized")
    except Exception as e:
        print(f"✗ Failed to initialize mixer: {e}")
        return
    
    # Create robot_sounds folder if it doesn't exist
    sound_folder = Path("robot_sounds")
    sound_folder.mkdir(exist_ok=True)
    
    # Check what sound files exist
    print(f"\n📁 Checking 'robot_sounds' folder...")
    sound_files = list(sound_folder.glob("*"))
    
    if not sound_files:
        print("No sound files found. Creating a test beep sound...")
        create_test_sounds(sound_folder)
    else:
        print(f"Found {len(sound_files)} sound file(s):")
        for file in sound_files:
            print(f"  - {file.name}")
    
    # Test 1: Generate a test beep sound
    print("\n🎵 TEST 1: Playing generated beep sound...")
    play_beep()
    time.sleep(1)
    
    # Test 2: Try to play existing sound files
    print("\n🎵 TEST 2: Trying to play existing sound files...")
    
    # Try common file types
    test_extensions = ['.mp3', '.wav', '.ogg']
    sound_played = False
    
    for ext in test_extensions:
        test_files = list(sound_folder.glob(f"*{ext}"))
        for sound_file in test_files[:2]:  # Try first 2 files of each type
            try:
                print(f"  Trying to play: {sound_file.name}")
                
                if sound_file.suffix.lower() == '.mp3':
                    # Play MP3
                    pygame.mixer.music.load(str(sound_file))
                    pygame.mixer.music.play()
                    print(f"    Playing MP3...")
                    
                    # Wait for it to finish
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                        
                else:
                    # Play WAV or OGG
                    sound = pygame.mixer.Sound(str(sound_file))
                    sound.play()
                    print(f"    Playing {sound_file.suffix}...")
                    
                    # Wait for sound to finish
                    time.sleep(min(sound.get_length(), 3))
                
                sound_played = True
                print(f"    ✓ Success!")
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
    
    if not sound_played:
        print("  No playable sound files found.")
    
    # Test 3: Volume control test
    print("\n🎚️ TEST 3: Testing volume control...")
    for volume in [0.3, 0.7, 1.0]:
        pygame.mixer.music.set_volume(volume)
        print(f"  Setting volume to {volume*100}%")
        play_beep(frequency=660 if volume == 0.3 else 880 if volume == 0.7 else 440)
        time.sleep(0.5)
    
    # Test 4: Multiple sounds test
    print("\n🎶 TEST 4: Testing multiple sounds...")
    print("  Playing three beeps in sequence:")
    for i, freq in enumerate([440, 550, 660]):
        print(f"    Beep {i+1} ({freq} Hz)")
        play_beep(frequency=freq, duration=0.2)
        time.sleep(0.3)
    
    print("\n" + "=" * 40)
    print("✅ Audio test complete!")
    print("\n💡 Next steps:")
    print("   1. Add your own MP3/WAV files to 'robot_sounds/' folder")
    print("   2. Run the full pizza robot simulation")
    print("   3. Connect external speakers via AUX if needed")
    
    pygame.mixer.quit()

def play_beep(frequency=440, duration=0.3, volume=0.5):
    """Generate and play a simple beep sound"""
    try:
        sample_rate = 22050
        n_samples = int(sample_rate * duration)
        
        # Generate sine wave
        import numpy as np
        samples = (32767 * volume * np.sin(2 * np.pi * np.arange(n_samples) * frequency / sample_rate)).astype(np.int16)
        
        # Play the sound
        sound = pygame.sndarray.make_sound(samples)
        sound.play()
        
        # Wait for sound to finish
        time.sleep(duration)
        
    except Exception as e:
        print(f"    Could not generate beep: {e}")

def create_test_sounds(sound_folder):
    """Create simple test sound files"""
    try:
        import numpy as np
        
        sample_rate = 22050
        
        # Create a test WAV file
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Generate a chord
        freq1 = 440  # A
        freq2 = 554  # C#
        freq3 = 659  # E
        
        samples = 0.6 * np.sin(2 * np.pi * freq1 * t)
        samples += 0.4 * np.sin(2 * np.pi * freq2 * t)
        samples += 0.3 * np.sin(2 * np.pi * freq3 * t)
        
        # Normalize to 16-bit range
        samples = (samples * 32767).astype(np.int16)
        
        # Save as WAV file
        import wave
        import struct
        
        with wave.open(str(sound_folder / "test.wav"), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for sample in samples:
                wav_file.writeframes(struct.pack('<h', sample))
        
        print(f"✓ Created test.wav in {sound_folder}")
        
    except Exception as e:
        print(f"Could not create test sound file: {e}")

def quick_pizza_simulation():
    """Run a quick pizza simulation with your laptop speakers"""
    
    print("\n" + "="*50)
    print("🍕 QUICK PIZZA ROBOT SIMULATION")
    print("="*50)
    
    # Initialize
    pygame.mixer.init()
    
    print("\n🤖 ROBOT: Welcome to PizzaBot! Testing audio system...")
    play_beep(523, 0.2)  # C note
    time.sleep(0.5)
    
    print("🤖 ROBOT: Pepperoni pizza coming up!")
    play_beep(587, 0.2)  # D note
    play_beep(659, 0.2)  # E note
    time.sleep(0.5)
    
    print("🤖 ROBOT: Hawaiian pizza - sweet and savory!")
    play_beep(698, 0.2)  # F note
    play_beep(784, 0.2)  # G note
    time.sleep(0.5)
    
    print("🤖 ROBOT: Order complete! Enjoy your meal!")
    play_beep(880, 0.3)  # A note
    play_beep(784, 0.2)  # G note
    play_beep(659, 0.4)  # E note
    
    print("\n✅ Simulation complete!")
    print("\n🔊 If you heard the beeps, your laptop speakers are working!")
    
    pygame.mixer.quit()

def check_system_audio():
    """Check if system audio is working"""
    print("\n🔧 System Audio Check")
    print("=" * 40)
    
    # Check OS
    import platform
    system = platform.system()
    print(f"Operating System: {system}")
    
    # Check if speakers are likely available
    if system == "Darwin":  # macOS
        print("macOS detected: Built-in speakers should work")
    elif system == "Windows":
        print("Windows detected: Check volume mixer")
    elif system == "Linux":
        print("Linux detected: Check pulseaudio/alsa")
    
    # Simple volume check reminder
    print("\n⚠️  Please ensure:")
    print("   1. Your laptop volume is not muted")
    print("   2. Volume is turned up (try 50% or higher)")
    print("   3. No headphones are plugged in (unless you want to use them)")

# Main menu
def main():
    print("🔊 Pizza Robot Audio Tester")
    print("=" * 40)
    print("Test the audio system on your laptop before using with DOFBot")
    print()
    print("Choose a test option:")
    print("  1. Basic speaker test")
    print("  2. Quick pizza simulation (with beeps)")
    print("  3. System audio check")
    print("  4. Run full pizza robot (if you have sound files)")
    print("  5. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                test_laptop_speaker()
            elif choice == "2":
                quick_pizza_simulation()
            elif choice == "3":
                check_system_audio()
            elif choice == "4":
                # Import and run the full robot
                print("\nRunning full pizza robot...")
                print("Make sure you have sound files in 'robot_sounds/' folder")
                
                # You'll need to have the full PizzaRobotAudio class defined
                try:
                    # Create a minimal version if the full class isn't available
                    print("Testing with minimal setup...")
                    pygame.mixer.init()
                    
                    # Check for sound files
                    sound_folder = Path("robot_sounds")
                    if sound_folder.exists() and any(sound_folder.iterdir()):
                        print("Sound files found! Playing greeting if available...")
                        
                        # Try to play any greeting sound
                        for file in sound_folder.glob("greeting.*"):
                            try:
                                if file.suffix.lower() == '.mp3':
                                    pygame.mixer.music.load(str(file))
                                    pygame.mixer.music.play()
                                    print(f"Playing {file.name}...")
                                    while pygame.mixer.music.get_busy():
                                        time.sleep(0.1)
                                else:
                                    sound = pygame.mixer.Sound(str(file))
                                    sound.play()
                                    time.sleep(min(sound.get_length(), 3))
                                break
                            except:
                                continue
                    else:
                        print("No sound files found. Run option 1 first to create test files.")
                    
                    pygame.mixer.quit()
                    
                except Exception as e:
                    print(f"Error: {e}")
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Please enter 1-5")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Run the interactive menu
    main()