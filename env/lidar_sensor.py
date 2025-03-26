import pybullet as p
import numpy as np
import math

class LidarSensor:
    """LIDAR sensor class for quadruped robots"""
    
    def __init__(self, robot_id, lidar_link_index=None, num_rays=36, ray_length=3.0, ray_start_length=0.1, 
                 ray_angle_start=0, ray_angle_end=2*math.pi, # Full 360 degrees from 0 to 2π
                 ray_color=[0, 1, 0], ray_hit_color=[1, 0, 0], x_offset=0.05, height_offset=0.02):
        """
        Args:
            robot_id: Robot ID
            lidar_link_index: Index of the link where LIDAR sensor is attached (Base link if None)
            num_rays: Number of rays
            ray_length: Maximum length of rays
            ray_start_length: Starting distance of rays (minimum distance from robot center)
            ray_angle_start: Starting angle of rays (radians)
            ray_angle_end: Ending angle of rays (radians)
            ray_color: Default ray color [R, G, B]
            ray_hit_color: Ray color when collision detected [R, G, B]
            x_offset: Offset in X-axis direction
            height_offset: Height offset of LIDAR from ground
        """
        self.robot_id = robot_id
        self.lidar_link_index = lidar_link_index if lidar_link_index is not None else -1  # -1 is base link
        self.num_rays = num_rays
        self.ray_length = ray_length
        self.ray_start_length = ray_start_length
        self.ray_angle_start = ray_angle_start
        self.ray_angle_end = ray_angle_end
        self.ray_color = ray_color
        self.ray_hit_color = ray_hit_color
        self.x_offset = x_offset
        self.height_offset = height_offset
        
        # Ray settings
        self.rayFrom = []
        self.rayTo = []
        self.rayIds = []
        
        # Direction area definitions (based on angle range)
        # Angle is -π/2 (-90 degrees) as forward, increasing counter-clockwise
        # Apply 90 degree angle adjustment
        self.angle_offset = math.pi/2  # 90 degree offset
        
        self.direction_ranges = {
            'front': (-math.pi/6, math.pi/6),       # Front ±30 degrees
            'left_side': (math.pi/6, math.pi/2),    # Left side 30 to 90 degrees
            'right_side': (-math.pi/2, -math.pi/6)  # Right side -90 to -30 degrees
        }
        
        # Define colors for each direction
        self.direction_colors = {
            'front': [0, 1, 0],      # Green (front)
            'left_side': [0, 1, 0],  # Green (left side)
            'right_side': [0, 1, 0]  # Green (right side)
        }
        
        # Define colors for when ray hits for each direction
        self.direction_hit_colors = {
            'front': [1, 0, 0],  # Red (front)
            'left_side': [1, 0, 0],  # Red (left side)
            'right_side': [1, 0, 0]  # Red (right side)
        }
        
        # Initialize rays
        self._initialize_rays()
    
    def _initialize_rays(self):
        """Initialize ray settings (full 360 degrees)"""
        self.rayFrom = []
        self.rayTo = []
        self.rayIds = []
        
        angle_range = self.ray_angle_end - self.ray_angle_start
        
        # Create mapping of which direction each ray belongs to based on its angle
        self.ray_direction_map = {}
        
        for i in range(self.num_rays):
            # Calculate angle (equally spaced around full 360 degree circle)
            angle = self.ray_angle_start + angle_range * float(i) / self.num_rays
            
            # Apply 90 degree (π/2) offset to adjust forward direction correctly
            adjusted_angle = angle + self.angle_offset
            
            # Normalize to range -π to π (with 0 as forward)
            normalized_angle = (adjusted_angle + math.pi) % (2 * math.pi) - math.pi
            
            # Determine direction
            ray_direction = None
            for direction, (min_angle, max_angle) in self.direction_ranges.items():
                if min_angle <= normalized_angle <= max_angle:
                    ray_direction = direction
                    break
            
            self.ray_direction_map[i] = ray_direction
            
            # Calculate X,Y components of ray (using unadjusted angle)
            rayX = math.sin(angle)
            rayY = math.cos(angle)
            
            # Set ray start and end points
            self.rayFrom.append([self.ray_start_length * rayX - self.x_offset, 
                                self.ray_start_length * rayY, 
                                self.height_offset])
            self.rayTo.append([self.ray_length * rayX, 
                              self.ray_length * rayY, 
                              self.height_offset])
            
            # Set color based on direction
            ray_color = self.ray_color  # Default color
            if ray_direction in self.direction_colors:
                ray_color = self.direction_colors[ray_direction]
            
            # Add ray for debug visualization (initially added in world coordinates)
            self.rayIds.append(p.addUserDebugLine(
                [0, 0, 0],  # Dummy start point (will be updated in scan)
                [0, 0, 0],  # Dummy end point (will be updated in scan)
                ray_color,  # Color based on direction
                lineWidth=2.0 if ray_direction else 1.0  # Thicker for specific directions
            ))
    
    def scan(self):
        """Execute LIDAR scan and return distances to obstacles"""
        distances = np.zeros(self.num_rays)
        hit_points = []
        
        # Get current robot position and orientation
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot_id)
        
        # Get LIDAR link local -> global transformation matrix
        if self.lidar_link_index == -1:
            # For base link
            lidar_pos = base_pos
            lidar_orient = base_orient
        else:
            # For other links
            lidar_state = p.getLinkState(self.robot_id, self.lidar_link_index)
            lidar_pos = lidar_state[0]  # Link position in world coordinates
            lidar_orient = lidar_state[1]  # Link orientation in world coordinates
        
        # Calculate ray start and end points in global coordinates
        rayFromWorld = []
        rayToWorld = []
        
        for i in range(self.num_rays):
            # Ray start point (local)
            ray_from_local = self.rayFrom[i]
            # Ray end point (local)
            ray_to_local = self.rayTo[i]
            
            # Convert local coordinates to global coordinates
            ray_from_world = p.multiplyTransforms(lidar_pos, lidar_orient, 
                                                 ray_from_local, [0, 0, 0, 1])[0]
            ray_to_world = p.multiplyTransforms(lidar_pos, lidar_orient, 
                                               ray_to_local, [0, 0, 0, 1])[0]
            
            rayFromWorld.append(ray_from_world)
            rayToWorld.append(ray_to_world)
        
        # Execute ray casting in batch
        results = p.rayTestBatch(rayFromWorld, rayToWorld)
        
        # Process results
        for i in range(self.num_rays):
            hit_object_uid = results[i][0]
            hit_fraction = results[i][2]
            hit_position = results[i][3]
            
            # Get ray direction
            ray_direction = self.ray_direction_map.get(i)
            
            # If no object hit or hit itself
            if hit_object_uid < 0 or hit_object_uid == self.robot_id:
                # No collision with objects
                ray_color = self.ray_color  # Default color
                if ray_direction in self.direction_colors:
                    ray_color = self.direction_colors[ray_direction]
                
                # Draw directly in world coordinates
                p.addUserDebugLine(
                    rayFromWorld[i], 
                    rayToWorld[i], 
                    ray_color,
                    lineWidth=2.0 if ray_direction else 1.0,
                    replaceItemUniqueId=self.rayIds[i]
                )
                distances[i] = self.ray_length
            else:
                # Ray hit an obstacle
                ray_hit_color = self.ray_hit_color  # Default hit color
                if ray_direction in self.direction_hit_colors:
                    ray_hit_color = self.direction_hit_colors[ray_direction]
                
                # Draw directly in world coordinates
                p.addUserDebugLine(
                    rayFromWorld[i], 
                    hit_position, 
                    ray_hit_color,
                    lineWidth=2.0 if ray_direction else 1.0,
                    replaceItemUniqueId=self.rayIds[i]
                )
                
                # Calculate distance from robot position
                dx = hit_position[0] - lidar_pos[0]
                dy = hit_position[1] - lidar_pos[1]
                distances[i] = math.sqrt(dx**2 + dy**2)
                
                # Record collision information
                hit_points.append({
                    'ray_index': i,
                    'direction': ray_direction,
                    'distance': distances[i],
                    'position': hit_position,
                    'object_id': hit_object_uid
                })
        
        return {
            'distances': distances,
            'hit_points': hit_points,
            'num_hits': len(hit_points)
        }
    
    # Remaining methods unchanged
    def update_visualization(self, show=True):
        """Update LIDAR visualization"""
        if not show:
            # Hide visualization
            for i in range(self.num_rays):
                p.removeUserDebugItem(self.rayIds[i])
            self.rayIds = []
        else:
            # If visualization is hidden, show it again
            if len(self.rayIds) == 0:
                self._initialize_rays()
    
    def get_closest_obstacle_direction(self, scan_result=None, min_distance=0.5):
        """
        Get the direction of the closest obstacle
        
        Args:
            scan_result: Scan result (if None, perform a new scan)
            min_distance: Minimum distance threshold for detection
            
        Returns:
            direction: Direction of closest obstacle (-1.0 to 1.0, -1.0 is left, 1.0 is right)
            distance: Distance to closest obstacle
            index: Corresponding ray index
            angle: Angle to obstacle (in radians, 0 is forward relative to robot)
        """
        if scan_result is None:
            scan_result = self.scan()
        
        distances = scan_result['distances']
        
        # Find obstacles below distance threshold
        obstacle_indices = np.where(distances < min_distance)[0]
        
        if len(obstacle_indices) == 0:
            # No obstacles
            return 0.0, float('inf'), -1, 0.0
        
        # Find closest obstacle
        closest_idx = obstacle_indices[np.argmin(distances[obstacle_indices])]
        closest_distance = distances[closest_idx]
        
        # Calculate angle (considering offset)
        angle = self.ray_angle_start + (self.ray_angle_end - self.ray_angle_start) * float(closest_idx) / self.num_rays
        adjusted_angle = angle + self.angle_offset
        
        # Calculate actual direction (-1.0 to 1.0)
        # 0 is forward, positive is right, negative is left
        angle_normalized = (adjusted_angle + math.pi) % (2 * math.pi) - math.pi  # Normalize to range -π to π
        normalized_direction = angle_normalized / math.pi  # Convert to range -1 to 1
        
        return normalized_direction, closest_distance, closest_idx, adjusted_angle
    
    def get_sector_analysis(self, scan_result=None, num_sectors=8):
        """
        Analyze LIDAR scan results by dividing into sectors (fan-shaped regions)
        
        Args:
            scan_result: Scan result (if None, perform a new scan)
            num_sectors: Number of sectors to divide into (default is 8 sectors = every 45 degrees)
            
        Returns:
            Dictionary of analysis results: minimum and average distance for each sector
        """
        if scan_result is None:
            scan_result = self.scan()
        
        distances = scan_result['distances']
        
        # Divide rays into sectors (evenly divide the full 360 degrees)
        rays_per_sector = self.num_rays // num_sectors
        sectors = []
        
        for i in range(num_sectors):
            start_idx = i * rays_per_sector
            end_idx = (i + 1) * rays_per_sector if i < num_sectors - 1 else self.num_rays
            
            sector_distances = distances[start_idx:end_idx]
            
            # Calculate center angle of sector (considering offset)
            start_angle = self.ray_angle_start + (self.ray_angle_end - self.ray_angle_start) * float(start_idx) / self.num_rays
            end_angle = self.ray_angle_start + (self.ray_angle_end - self.ray_angle_start) * float(end_idx) / self.num_rays
            center_angle = (start_angle + end_angle) / 2.0 + self.angle_offset
            
            # Add sector identifier (front, front-right, right, back-right, back, back-left, left, front-left)
            # Need to rotate considering the offset
            rotated_sector_index = (i + int(num_sectors / 4)) % num_sectors  # Equivalent to 90 degree (π/2) offset
            sector_names = ['front', 'front_right', 'right', 'back_right', 'back', 'back_left', 'left', 'front_left']
            sector_name = sector_names[rotated_sector_index % len(sector_names)]
            
            sectors.append({
                'min_distance': np.min(sector_distances),
                'mean_distance': np.mean(sector_distances),
                'ray_indices': list(range(start_idx, end_idx)),
                'center_angle': center_angle,
                'name': sector_name
            })
        
        return sectors
    
    def get_direction_distances(self, scan_result=None):
        """
        Get distance information for front, left side, and right side
        
        Args:
            scan_result: Scan result (if None, perform a new scan)
            
        Returns:
            dict: Minimum and average distances for each direction
        """
        if scan_result is None:
            scan_result = self.scan()
        
        distances = scan_result['distances']
        direction_distances = {}
        
        # Group rays by direction
        direction_rays = {direction: [] for direction in self.direction_ranges.keys()}
        
        for i in range(self.num_rays):
            ray_direction = self.ray_direction_map.get(i)
            if ray_direction:
                direction_rays[ray_direction].append(i)
        
        # Calculate distance statistics for each direction
        for direction, indices in direction_rays.items():
            if indices:
                direction_distances[direction] = {
                    'min_distance': np.min(distances[indices]),
                    'mean_distance': np.mean(distances[indices]),
                    'ray_indices': indices
                }
            else:
                direction_distances[direction] = {
                    'min_distance': self.ray_length,
                    'mean_distance': self.ray_length,
                    'ray_indices': []
                }
        
        return direction_distances