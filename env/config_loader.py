import yaml
import os
from pathlib import Path

class ConfigLoader:
    """
    Class for loading and managing configurations from YAML files
    """
    
    def __init__(self, config_path=None):
        """
        Args:
            config_path (str, optional): Path to the configuration file. Default settings will be used if None.
        """
        self.config = {}
        
        # Load default configuration
        self._load_default_config()
        
        # Override with specified file if it exists
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
            
    def _load_default_config(self):
        """Load default configuration values"""
        self.config = {
            # Robot settings
            'robot': {
                'urdf_path': 'models/urdf/svdog2_2_description/svdog2_2.urdf',
                'position': [0, 0, 0.08],
                'rotation': [0, 0, 135],
                'max_force': 5.0
            },
            
            # Environment settings
            'environment': {
                'use_gui': True,
                'camera_follow': True,
                'gravity': [0, 0, -9.8],
                'timestep': 0.00416667  # 1/240
            },
            
            # LiDAR settings
            'lidar': {
                'enabled': True,
                'num_rays': 36,
                'ray_length': 1.0,
                'ray_start_length': 0.01,
                'ray_color': [0, 1, 0],
                'ray_hit_color': [1, 0, 0]
            },
            
            # Gait generator settings
            'gait': {
                'amplitude': 0.25,
                'frequency': 1.5,
                'pattern': 'trot',
                'turn_direction': 0,
                'turn_intensity': 0
            },
            
            # Obstacle settings
            'obstacles': {
                'enabled': True,
                'course_type': 'simple',
                'length': 5.0
            },
            
            # Goal settings
            'goal': {
                'enabled': True,
                'position': [2.0, 0, 0],
                'radius': 0.3,
                'color': [0.0, 0.8, 0.0, 0.5]
            },
            
            # Simulation settings
            'simulation': {
                'max_steps': 5000,
                'debug_interval': 100
            }
        }
        
    def load_config(self, config_path):
        """
        Load configuration from the specified YAML file
        
        Args:
            config_path (str): Path to the configuration file
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                user_config = yaml.safe_load(file)
                
            # Deep update of default configuration with loaded settings
            self._deep_update(self.config, user_config)
            print(f"Configuration loaded from '{config_path}'")
            
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            
    def _deep_update(self, original, update):
        """
        Recursively update a dictionary
        
        Args:
            original (dict): Original dictionary to be updated
            update (dict): Dictionary containing update values
        """
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
                
    def get(self, section, key=None, default=None):
        """
        Get a configuration value
        
        Args:
            section (str): Configuration section name
            key (str, optional): Configuration key. Returns the entire section if None
            default: Default value to return if key doesn't exist
            
        Returns:
            Configuration value or default value
        """
        if section not in self.config:
            return default
            
        if key is None:
            return self.config[section]
            
        return self.config[section].get(key, default)
        
    def save_config(self, config_path):
        """
        Save current configuration to a YAML file
        
        Args:
            config_path (str): Destination file path
        """
        try:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(config_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
            with open(config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)
                
            print(f"Configuration saved to '{config_path}'")
            
        except Exception as e:
            print(f"Failed to save configuration file: {e}")
            
    def export_default_config(self, path):
        """
        Export default configuration to a YAML file
        
        Args:
            path (str): Destination file path
        """
        self._load_default_config()
        self.save_config(path)