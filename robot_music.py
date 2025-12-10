
import pygame
import os
from pathlib import Path
import logging

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
            pygame.mixer.music.set_volume(0.7)
            self.audio_initialized = True
            self.logger.info("Audio system initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize audio: {e}")
            self.audio_initialized = False
            return
        
        # Map of sound events to their sound files
        self.sound_mapping = {
            'greeting': ['greetings.wav'],
            'Anchovy Special': ['anchovy.wav'],
            'Pesto Chicken': ['pesto.wav'],
            'Ocean Garden': ['ocean_garden.wav'],
            'Chicken Supreme': ['chicken_supreme.wav'],
            'Margherita': ['margherita.wav'],
            'Seafood Delight': ['seafood.wav'],
            'goodbye': ['goodbye.wav'],
            'found_cube': ['found.wav'],
            'drop_cube': ['delivery_done.wav']
        
        }
        
        # Load all sounds
        self.sounds = {}
        self.sound_folder = Path(sound_folder)
        
        # Create sound folder if it doesn't exist
        self.sound_folder.mkdir(exist_ok=True)
        
        # Pre-load all available sounds
        self.load_sounds()
    
    def find_sound_file(self, sound_key):
        """
        Find the actual sound file for a given key
        Returns the path to the file or None if not found
        """
        if sound_key in self.sound_mapping:
            for filename in self.sound_mapping[sound_key]:
                sound_path = self.sound_folder / filename
                if sound_path.exists():
                    return sound_path
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
    
    def play_sound(self, sound_key):
        """
        Play a sound effect (MP3 or WAV) - non-blocking
        
        Args:
            sound_key: Key from sound_mapping (e.g., 'greeting', 'Chicken Supreme')
        
        Returns:
            bool: True if sound played successfully
        """
        if not self.audio_initialized:
            return False
            
        if sound_key in self.sounds and self.sounds[sound_key] is not None:
            try:
                sound_path = self.sounds[sound_key]
                
                # Play MP3 or WAV file
                if sound_path.lower().endswith('.mp3'):
                    pygame.mixer.music.load(sound_path)
                    pygame.mixer.music.play()
                else:
                    sound = pygame.mixer.Sound(sound_path)
                    sound.play()
                
                self.logger.info(f"Playing sound: {sound_key}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error playing sound {sound_key}: {e}")
                return False
        else:
            self.logger.warning(f"Sound not available: {sound_key}")
            return False
    
    # Convenience methods for common sounds
    def play_greeting(self):
        """Play greeting sound"""
        return self.play_sound('greeting')
    
    def play_goodbye(self):
        """Play goodbye sound"""
        return self.play_sound('goodbye')
    
    def play_pizza_sound(self, pizza_name):
        """Play sound for a specific pizza"""
        return self.play_sound(pizza_name)
    
    def play_found_cube(self):
        """Play sound when cube is found"""
        return self.play_sound('found_cube')
    
    def play_drop_cube(self):
        """Play sound when cube is dropped"""
        return self.play_sound('drop_cube')
    
   
    def stop_all_sounds(self):
        """Stop all currently playing sounds"""
        try:
            pygame.mixer.stop()
            pygame.mixer.music.stop()
            return True
        except Exception as e:
            self.logger.error(f"Error stopping sounds: {e}")
            return False
    
    def cleanup(self):
        """Cleanup audio resources"""
        try:
            self.stop_all_sounds()
            pygame.mixer.quit()
            self.logger.info("Audio system cleaned up")
        except:
            pass