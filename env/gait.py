import numpy as np

class GaitGenerator:
    """Gait pattern generator class for quadruped robots"""
    
    def __init__(self, amplitude=0.15, frequency=1.0):
        """
        Initialize the gait generator
        
        Args:
            amplitude: Joint angle amplitude
            frequency: Walking cycle frequency
        """
        self.time_step = 0
        self.num_legs = 4
        self.num_joints_per_leg = 2
        self.amplitude = amplitude
        self.frequency = frequency
        
        # Joint mapping for quadruped robot
        self.joint_map = {
            'LF_upper': 4, 'LF_lower': 5,  # Left Front leg
            'RF_upper': 2, 'RF_lower': 3,  # Right Front leg
            'LH_upper': 6, 'LH_lower': 7,  # Left Hind leg
            'RH_upper': 0, 'RH_lower': 1   # Right Hind leg
        }
        
        # Base joint angles
        self.knee_base_angle = 0.3   # Knee base angle
        self.hip_base_angle = -0.3   # Hip base angle
        
        # Direction control parameters
        self.turn_direction = 0.0    # -1.0 (left) to 1.0 (right)
        self.turn_intensity = 0.1    # Turning intensity
        self.moving_backward = False  # Backward movement flag
        self.backward_intensity = 1.0 # Backward movement intensity
        
        # Leg step height adjustment
        self.step_height_multiplier = 1.2  # Height adjustment for inner legs
        
        # Define available gait patterns
        self.gait_patterns = {
            'trot': self._create_trot_phases(),
            'walk': self._create_walk_phases(),
            'bound': self._create_bound_phases()
        }
        
        # Set default pattern
        self.current_pattern = 'trot'
    
    def set_backward(self, enable=True, intensity=1.0):
        """
        Set backward movement
        Args:
            enable (bool): Enable backward movement
            intensity (float): Backward movement intensity (0.0 to 1.0)
        """
        self.moving_backward = enable
        self.backward_intensity = np.clip(intensity, 0.0, 1.0)
        print(f"Movement direction: {'Backward' if enable else 'Forward'}, Intensity: {self.backward_intensity}")

    def set_gait_pattern(self, pattern):
        """Set gait pattern"""
        if pattern in self.gait_patterns:
            self.current_pattern = pattern
            print(f"Gait pattern set to {pattern}")
        else:
            available_patterns = list(self.gait_patterns.keys())
            raise ValueError(f"Unknown gait pattern: {pattern}. Available patterns: {available_patterns}")
    
    def set_turn_direction(self, direction):
        """Set turn direction (-1.0 for left, 1.0 for right)"""
        self.turn_direction = np.clip(direction, -1.0, 1.0)
        turn_str = "Left" if direction < 0 else "Right" if direction > 0 else "Straight"
        # print(f"Turn direction set to {turn_str} ({self.turn_direction})")
    
    def set_turn_intensity(self, intensity):
        """Set turn intensity (0.0 to 1.0)"""
        self.turn_intensity = np.clip(intensity, 0.0, 1.0)
        # print(f"Turn intensity set to {self.turn_intensity}")
    
    def _create_trot_phases(self):
        """Create phase offsets for trot gait"""
        return {
            'LF_upper': 0,          
            'LF_lower': np.pi / 2,
            'RH_upper': np.pi,   
            'RH_lower': 3 * np.pi / 2,
            
            'RF_upper': 0,  
            'RF_lower': np.pi / 2,  
            'LH_upper': np.pi,     
            'LH_lower': 3 * np.pi / 2 
        }
    
    def _create_walk_phases(self):
        """Create phase offsets for walk gait"""
        return {
            'LF_upper': 0,          
            'LF_lower': np.pi / 2,
            'RF_upper': np.pi / 2,   
            'RF_lower': np.pi,
            'LH_upper': 3 * np.pi / 2,     
            'LH_lower': 2 * np.pi,
            'RH_upper': np.pi,   
            'RH_lower': 3 * np.pi / 2
        }
    
    def _create_bound_phases(self):
        """Create phase offsets for bound gait"""
        return {
            'LF_upper': 0,          
            'LF_lower': np.pi / 2,
            'RF_upper': 0,   
            'RF_lower': np.pi / 2,
            'LH_upper': np.pi,     
            'LH_lower': 3 * np.pi / 2,
            'RH_upper': np.pi,   
            'RH_lower': 3 * np.pi / 2
        }
    
    def _apply_turn_adjustment(self, joint_name, base_angle, amplitude_mult=1.0):
        """Apply turn adjustment to base angle and adjust amplitude"""
        turn_offset = 0.0
        
        # Determine left/right side
        is_right_side = 'R' in joint_name
        is_front = 'F' in joint_name
        is_upper = 'upper' in joint_name
        
        # Adjustment based on turn direction
        if self.turn_direction != 0:
            # Right turn
            if self.turn_direction > 0:
                if not is_right_side:  # Left side legs
                    if is_upper:
                        turn_offset = self.turn_intensity * 0.4
                        amplitude_mult *= self.step_height_multiplier
                    else:
                        turn_offset = -self.turn_intensity * 0.2
            else:  # Left turn
                if is_right_side:  # Right side legs
                    if is_upper:
                        turn_offset = -self.turn_intensity * 0.4
                        amplitude_mult *= self.step_height_multiplier
                    else:
                        turn_offset = self.turn_intensity * 0.2
            
            # Different adjustment for front and hind legs
            if is_front:
                turn_offset *= 1.2
        
        # Adjustment for backward movement
        if self.moving_backward and is_upper:
            # Different backward adjustment for front and hind legs
            backward_offset = 0.3 * self.backward_intensity
            if is_front:
                base_angle = -base_angle  # Reverse movement for front legs
                turn_offset *= -1  # Also reverse turn direction
            else:
                base_angle = -base_angle  # Also reverse movement for hind legs
                turn_offset *= -1
        
        adjusted_angle = base_angle + turn_offset * abs(self.turn_direction)
        return adjusted_angle, amplitude_mult
    
    def get_action(self):
        """Calculate joint angles for the current time step"""
        time_param = 2 * np.pi * self.frequency * self.time_step
        if self.moving_backward:
            time_param = -time_param  # Reverse phase to achieve backward movement
        
        action = np.zeros(8)
        phases = self.gait_patterns[self.current_pattern]
        
        # Calculate angle for each joint
        for joint_name, idx in self.joint_map.items():
            phase = phases[joint_name]
            
            # Adjust base angle based on joint type and side
            if 'lower' in joint_name:  # Knee joint
                if 'R' in joint_name:  # Right leg
                    base_angle = -self.knee_base_angle
                else:  # Left leg
                    base_angle = self.knee_base_angle
            else:  # Hip joint
                if 'R' in joint_name:  # Right leg
                    base_angle = -self.hip_base_angle
                else:  # Left leg
                    base_angle = self.hip_base_angle
            
            # Apply turn adjustment and get modified amplitude
            base_angle, amplitude_mult = self._apply_turn_adjustment(joint_name, base_angle)
            
            # Calculate final angle using sine wave pattern with adjusted amplitude
            action[idx] = base_angle + (self.amplitude * amplitude_mult) * np.sin(time_param + phase)
        
        self.time_step += 0.01
        return action