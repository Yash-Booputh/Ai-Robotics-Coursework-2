
import pygame
import os
from pathlib import Path
import logging
import random

class PizzaRobotAudio:
    def __init__(self, sound_folder="robot_sounds"):
        """
        Initialize the pizza robot audio system

        Args:
            sound_folder: Path to folder containing sound effects
        """
        self.logger = logging.getLogger(__name__)

        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
            self.audio_initialized = True
            self.logger.info("Audio system initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize audio: {e}")
            self.audio_initialized = False
            return

        # Volume settings for different sound types
        self.volumes = {
            'background': 0.25,      # Ambient background music
            'greeting': 0.7,         # Welcome sound
            'menu': 0.75,            # Pizza menu sounds
            'found': 0.6,            # Ingredient found
            'delivery': 0.7,         # Delivery/drop sounds
            'error': 0.8             # Error sounds
        }

        # Background music state
        self.background_music_playing = False

        # Map of sound events to their sound files
        self.sound_mapping = {
            'greeting': ['greetings.wav'],
            'background_music': ['background_music.mp3'],
            'Anchovy Special': ['anchovy.wav'],
            'Pesto Chicken': ['pesto.wav'],
            'Ocean Garden': ['ocean_garden.wav'],
            'Chicken Supreme': ['chicken_supreme.wav'],
            'Margherita': ['margherita.wav'],
            'Seafood Delight': ['seafood.wav'],
            'found_cube': ['found.wav'],
            'drop_cube': ['delivery_done.wav', 'delivery_done.wav2.wav'],  # Random selection
            'error': ['error_1.wav']
        }
        
        # Load all sounds
        self.sounds = {}
        self.sound_folder = Path(sound_folder)
        
        # Create sound folder if it doesn't exist
        self.sound_folder.mkdir(exist_ok=True)
        
        # Pre-load all available sounds
        self.load_sounds()
    
    def find_sound_file(self, sound_key, select_random=False):
        """
        Find the actual sound file for a given key
        Returns the path to the file or None if not found

        Args:
            sound_key: Key from sound_mapping
            select_random: If True and multiple files available, pick one randomly
        """
        if sound_key in self.sound_mapping:
            files = self.sound_mapping[sound_key]

            # Filter to only existing files
            existing_files = []
            for filename in files:
                sound_path = self.sound_folder / filename
                if sound_path.exists():
                    existing_files.append(sound_path)

            if existing_files:
                if select_random and len(existing_files) > 1:
                    return random.choice(existing_files)
                else:
                    return existing_files[0]

        return None
    
    def load_sounds(self):
        """Try to load all available sound effects from the sound folder"""
        for key in self.sound_mapping.keys():
            sound_path = self.find_sound_file(key)
            if sound_path:
                try:
                    self.sounds[key] = str(sound_path)
                    self.logger.info(f"Loaded sound: {sound_path.name}")
                except Exception as e:
                    self.logger.error(f"Error with {sound_path.name}: {e}")
                    self.sounds[key] = None
            else:
                self.logger.warning(f"Sound file not found for {key}")
                self.sounds[key] = None
    
    def play_sound(self, sound_key, volume_type='menu', select_random=False):
        """
        Play a sound effect (MP3 or WAV) - non-blocking

        Args:
            sound_key: Key from sound_mapping (e.g., 'greeting', 'Chicken Supreme')
            volume_type: Type of volume to use ('background', 'greeting', 'menu', 'found', 'delivery', 'error')
            select_random: If True, randomly select from multiple files (for drop_cube)

        Returns:
            bool: True if sound played successfully
        """
        if not self.audio_initialized:
            return False

        if sound_key in self.sounds and self.sounds[sound_key] is not None:
            try:
                # Get sound path (potentially random for drop_cube)
                if select_random and sound_key == 'drop_cube':
                    sound_path = self.find_sound_file(sound_key, select_random=True)
                    if sound_path:
                        sound_path = str(sound_path)
                    else:
                        sound_path = self.sounds[sound_key]
                else:
                    sound_path = self.sounds[sound_key]

                # Get volume for this sound type
                volume = self.volumes.get(volume_type, 0.7)

                # Play MP3 or WAV file
                if sound_path.lower().endswith('.mp3'):
                    # Use music channel for MP3 (only for non-background music)
                    if sound_key != 'background_music':
                        pygame.mixer.music.load(sound_path)
                        pygame.mixer.music.set_volume(volume)
                        pygame.mixer.music.play()
                else:
                    # Use Sound channel for WAV
                    sound = pygame.mixer.Sound(sound_path)
                    sound.set_volume(volume)
                    sound.play()

                self.logger.info(f"Playing sound: {sound_key} (volume: {volume})")
                return True

            except Exception as e:
                self.logger.error(f"Error playing sound {sound_key}: {e}")
                return False
        else:
            self.logger.warning(f"Sound not available: {sound_key}")
            return False
    
    # =========================================================================
    # BACKGROUND MUSIC METHODS
    # =========================================================================

    def start_background_music(self):
        """Start playing background music on loop"""
        if not self.audio_initialized:
            return False

        if self.background_music_playing:
            self.logger.info("Background music already playing")
            return True

        sound_path = self.find_sound_file('background_music')
        if sound_path:
            try:
                pygame.mixer.music.load(str(sound_path))
                pygame.mixer.music.set_volume(self.volumes['background'])
                pygame.mixer.music.play(-1)  # -1 means loop indefinitely
                self.background_music_playing = True
                self.logger.info("Background music started (looping)")
                return True
            except Exception as e:
                self.logger.error(f"Error starting background music: {e}")
                return False
        else:
            self.logger.warning("Background music file not found")
            return False

    def stop_background_music(self):
        """Stop background music"""
        try:
            pygame.mixer.music.stop()
            self.background_music_playing = False
            self.logger.info("Background music stopped")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping background music: {e}")
            return False

    def is_background_music_playing(self):
        """Check if background music is currently playing"""
        return self.background_music_playing and pygame.mixer.music.get_busy()

    def set_background_volume(self, volume):
        """
        Set the background music volume

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if not self.audio_initialized:
            return False

        try:
            # Clamp volume between 0.0 and 1.0
            volume = max(0.0, min(1.0, volume))

            # Update stored volume
            self.volumes['background'] = volume

            # Update pygame music volume if music is playing
            pygame.mixer.music.set_volume(volume)

            self.logger.info(f"Background music volume set to {volume:.2f}")
            return True
        except Exception as e:
            self.logger.error(f"Error setting background volume: {e}")
            return False

    # =========================================================================
    # CONVENIENCE METHODS FOR COMMON SOUNDS
    # =========================================================================

    def play_greeting(self):
        """Play greeting sound"""
        return self.play_sound('greeting', volume_type='greeting')

    def play_pizza_sound(self, pizza_name):
        """Play sound for a specific pizza"""
        return self.play_sound(pizza_name, volume_type='menu')

    def play_found_cube(self):
        """Play sound when cube is found"""
        return self.play_sound('found_cube', volume_type='found')

    def play_drop_cube(self):
        """Play sound when cube is dropped (randomly selects from available sounds)"""
        return self.play_sound('drop_cube', volume_type='delivery', select_random=True)

    def play_error(self):
        """Play error sound"""
        return self.play_sound('error', volume_type='error')

    def stop_all_sounds(self):
        """Stop all currently playing sounds (except background music)"""
        try:
            pygame.mixer.stop()  # Stop all Sound channels
            return True
        except Exception as e:
            self.logger.error(f"Error stopping sounds: {e}")
            return False
    
    def cleanup(self):
        """Cleanup audio resources"""
        try:
            self.stop_all_sounds()
            self.stop_background_music()
            pygame.mixer.quit()
            self.logger.info("Audio system cleaned up")
        except:
            pass