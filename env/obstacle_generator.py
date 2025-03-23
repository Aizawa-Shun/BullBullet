import pybullet as p
import numpy as np
import os

class ObstacleGenerator:
    """Obstacle generator class for quadruped robot simulation"""
    
    def __init__(self, env):
        """
        Initialize the obstacle generator class
        
        Args:
            env (RobotEnvironment): Instance of robot environment
        """
        self.env = env
        self.obstacles = []
        self.obstacle_types = {
            'box': self._create_box,
            'sphere': self._create_sphere,
            'cylinder': self._create_cylinder,
            'ramp': self._create_ramp,
            'step': self._create_step,
            'irregular': self._create_irregular_terrain
        }
    
    def _create_box(self, position, size=[0.1, 0.1, 0.1], mass=0.0, color=[0.8, 0.2, 0.2, 1.0]):
        """
        Create a box-shaped obstacle
        
        Args:
            position (list): Placement position [x, y, z]
            size (list): Size [width, depth, height]
            mass (float): Mass (0.0 for fixed obstacle)
            color (list): Color [r, g, b, a]
            
        Returns:
            int: Object ID of the created box
        """
        collision_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size])
        visual_box_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in size], rgbaColor=color)
        
        box_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_box_id,
            baseVisualShapeIndex=visual_box_id,
            basePosition=position
        )
        
        self.obstacles.append({
            'id': box_id,
            'type': 'box',
            'position': position,
            'size': size
        })
        
        return box_id
    
    def _create_sphere(self, position, radius=0.1, mass=0.0, color=[0.2, 0.2, 0.8, 1.0]):
        """
        Create a sphere-shaped obstacle
        
        Args:
            position (list): Placement position [x, y, z]
            radius (float): Sphere radius
            mass (float): Mass (0.0 for fixed obstacle)
            color (list): Color [r, g, b, a]
            
        Returns:
            int: Object ID of the created sphere
        """
        collision_sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visual_sphere_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        
        sphere_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_sphere_id,
            baseVisualShapeIndex=visual_sphere_id,
            basePosition=position
        )
        
        self.obstacles.append({
            'id': sphere_id,
            'type': 'sphere',
            'position': position,
            'radius': radius
        })
        
        return sphere_id
    
    def _create_cylinder(self, position, radius=0.1, height=0.2, mass=0.0, color=[0.2, 0.8, 0.2, 1.0]):
        """
        Create a cylinder-shaped obstacle
        
        Args:
            position (list): Placement position [x, y, z]
            radius (float): Cylinder radius
            height (float): Cylinder height
            mass (float): Mass (0.0 for fixed obstacle)
            color (list): Color [r, g, b, a]
            
        Returns:
            int: Object ID of the created cylinder
        """
        collision_cylinder_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        visual_cylinder_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        
        cylinder_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_cylinder_id,
            baseVisualShapeIndex=visual_cylinder_id,
            basePosition=position
        )
        
        self.obstacles.append({
            'id': cylinder_id,
            'type': 'cylinder',
            'position': position,
            'radius': radius,
            'height': height
        })
        
        return cylinder_id
    
    def _create_ramp(self, position, size=[0.5, 0.3, 0.1], angle=15, color=[0.5, 0.5, 0.5, 1.0]):
        """
        Create a ramp
        
        Args:
            position (list): Placement position [x, y, z]
            size (list): Size [length, width, height]
            angle (float): Incline angle (degrees)
            color (list): Color [r, g, b, a]
            
        Returns:
            int: Object ID of the created ramp
        """
        # Convert angle to radians
        angle_rad = angle * np.pi / 180.0
        
        # Calculate actual dimensions from ramp size
        length = size[0]
        width = size[1]
        height = size[2]
        
        # Create collision shape for the ramp
        collision_ramp_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2])
        visual_ramp_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2], rgbaColor=color)
        
        # Set orientation of the ramp
        orientation = p.getQuaternionFromEuler([angle_rad, 0, 0])
        
        # Adjust height so that the base of the ramp meets the ground
        adjusted_height = np.sin(angle_rad) * length / 2 + height / 2 * np.cos(angle_rad)
        adjusted_position = [position[0], position[1], position[2] + adjusted_height]
        
        ramp_id = p.createMultiBody(
            baseMass=0.0,  # Fixed obstacle
            baseCollisionShapeIndex=collision_ramp_id,
            baseVisualShapeIndex=visual_ramp_id,
            basePosition=adjusted_position,
            baseOrientation=orientation
        )
        
        self.obstacles.append({
            'id': ramp_id,
            'type': 'ramp',
            'position': adjusted_position,
            'size': size,
            'angle': angle
        })
        
        return ramp_id
    
    def _create_step(self, position, num_steps=5, step_width=0.2, step_length=0.15, step_height=0.006, color=[0.4, 0.4, 0.4, 1.0]):
        """
        Create stair-like obstacles
        
        Args:
            position (list): Starting position of the stairs [x, y, z]
            num_steps (int): Number of steps
            step_width (float): Width of each step
            step_length (float): Length of each step
            step_height (float): Height of each step
            color (list): Color [r, g, b, a]
            
        Returns:
            list: List of object IDs for each created step
        """
        step_ids = []
        
        for i in range(num_steps):
            # Calculate position for each step
            step_pos = [
                position[0] + i * step_length,  # Stack in X direction
                position[1],
                position[2] + i * step_height / 2  # Gradually increase height
            ]
            
            # Size of each step
            step_size = [step_length, step_width, (i + 1) * step_height]
            
            # Create the step
            step_id = self._create_box(step_pos, step_size, mass=0.0, color=color)
            step_ids.append(step_id)
        
        # Record all steps together
        self.obstacles.append({
            'id': step_ids,
            'type': 'step',
            'position': position,
            'num_steps': num_steps,
            'step_size': [step_length, step_width, step_height]
        })
        
        return step_ids
    
    def _create_irregular_terrain(self, position, size=[2.0, 2.0], resolution=0.1, height_range=0.05, smoothness=2.0, color=[0.6, 0.6, 0.3, 1.0]):
        """
        Create irregular terrain
        
        Args:
            position (list): Center position of the terrain [x, y, z]
            size (list): Size of the terrain [width, depth]
            resolution (float): Resolution of the terrain (mesh size)
            height_range (float): Range of height variations
            smoothness (float): Terrain smoothness (higher is smoother)
            color (list): Color [r, g, b, a]
            
        Returns:
            int: Object ID of the created terrain
        """
        # Calculate terrain grid size
        grid_size_x = int(size[0] / resolution)
        grid_size_y = int(size[1] / resolution)
        
        # Generate height map using Perlin noise (simplified by using random heights)
        height_field = np.zeros((grid_size_x, grid_size_y), dtype=np.float32)
        
        # Generate simple irregular terrain (ideally Perlin noise would be better)
        np.random.seed(42)  # For reproducibility
        for i in range(grid_size_x):
            for j in range(grid_size_y):
                # Simplified noise generation (Perlin noise would be better in practice)
                x_contrib = np.sin(i / smoothness) * height_range / 3
                y_contrib = np.cos(j / smoothness) * height_range / 3
                random_contrib = np.random.uniform(-height_range/3, height_range/3)
                height_field[i, j] = x_contrib + y_contrib + random_contrib + height_range/2
        
        # Normalize height map values to 0-1 range
        height_field = (height_field - np.min(height_field)) / (np.max(height_field) - np.min(height_field))
        
        # Convert height map to PyBullet format
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[resolution, resolution, height_range],
            heightfieldTextureScaling=(grid_size_x-1)/2,
            heightfieldData=height_field.flatten(),
            numHeightfieldRows=grid_size_x,
            numHeightfieldColumns=grid_size_y
        )
        
        # Adjust position so that the center of the height map is at the specified position
        adjusted_position = [
            position[0] - size[0]/2,
            position[1] - size[1]/2,
            position[2]
        ]
        
        terrain_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=terrain_shape,
            basePosition=adjusted_position
        )
        
        # Set terrain color
        p.changeVisualShape(terrain_id, -1, rgbaColor=color)
        
        self.obstacles.append({
            'id': terrain_id,
            'type': 'irregular',
            'position': position,
            'size': size,
            'resolution': resolution,
            'height_range': height_range
        })
        
        return terrain_id
    
    def create_obstacle_course(self, course_type='simple', start_position=[0, 0, 0], length=5.0):
        """
        Create a predefined obstacle course
        
        Args:
            course_type (str): Type of course ('simple', 'dense', 'random')
            start_position (list): Starting position of the course [x, y, z]
            length (float): Length of the course
            
        Returns:
            list: List of created obstacles
        """
        if course_type == 'simple':
            # Simple obstacle course
            # First obstacles (right in front of the robot)
            self._create_box([start_position[0] + 0.1, start_position[1] + 0.2, 0.05], [0.15, 0.15, 0.1])
            self._create_box([start_position[0] + 0.1, start_position[1] - 0.2, 0.05], [0.15, 0.15, 0.1])
            
            # Place stairs
            self._create_step([start_position[0] + 1.0, 0, 0.0], num_steps=3, step_width=0.4)
            
            # Add several additional obstacles
            self._create_sphere([start_position[0] + 2.0, start_position[1] + 0.25, 0.08], radius=0.08)
            self._create_sphere([start_position[0] + 2.0, start_position[1] - 0.25, 0.08], radius=0.08)
            self._create_cylinder([start_position[0] + 2.3, start_position[1], 0.1], radius=0.1, height=0.2)
            self._create_cylinder([start_position[0]+ 0.6, start_position[1], 0.08], radius=0.05, height=0.16, color=[0.2, 0.9, 0.2, 1.0])
            
            # Further place a ramp
            self._create_ramp([start_position[0] + 2.8, start_position[1], 0.0], size=[0.4, 0.6, 0.1], angle=15)
            
            # Final obstacles
            self._create_box([start_position[0] + 3.5, start_position[1] + 0.15, 0.06], [0.12, 0.12, 0.12])
            self._create_box([start_position[0] + 3.5, start_position[1] - 0.15, 0.06], [0.12, 0.12, 0.12])
            
            return self.obstacles
            
        elif course_type == 'dense':
            # Very dense obstacle course
            # Set basic spacing
            spacing = 0.25  # Basic spacing between obstacles
            
            # Place obstacles densely ahead
            for i in range(int(length / spacing)):
                x_pos = start_position[0] + i * spacing
                
                # Alternately place obstacles on left and right
                if i % 2 == 0:
                    # Obstacles on the right
                    y_offset = 0.15
                    obs_type = i % 3  # Vary obstacle type
                    
                    if obs_type == 0:
                        self._create_box([x_pos, start_position[1] + y_offset, 0.05], [0.1, 0.1, 0.1], color=[0.9, 0.2, 0.2, 1.0])
                    elif obs_type == 1:
                        self._create_sphere([x_pos, start_position[1] + y_offset, 0.06], radius=0.06, color=[0.2, 0.2, 0.9, 1.0])
                    else:
                        self._create_cylinder([x_pos, start_position[1] + y_offset, 0.08], radius=0.05, height=0.16, color=[0.2, 0.9, 0.2, 1.0])
                else:
                    # Obstacles on the left
                    y_offset = -0.15
                    obs_type = (i + 1) % 3  # Vary obstacle type
                    
                    if obs_type == 0:
                        self._create_box([x_pos, start_position[1] + y_offset, 0.05], [0.1, 0.1, 0.1], color=[0.9, 0.3, 0.3, 1.0])
                    elif obs_type == 1:
                        self._create_sphere([x_pos, start_position[1] + y_offset, 0.06], radius=0.06, color=[0.3, 0.3, 0.9, 1.0])
                    else:
                        self._create_cylinder([x_pos, start_position[1] + y_offset, 0.08], radius=0.05, height=0.16, color=[0.3, 0.9, 0.3, 1.0])
                
                # Add small obstacles in the center (every 5th)
                if i % 5 == 3:
                    self._create_sphere([x_pos + spacing/2, start_position[1], 0.05], radius=0.05, color=[0.9, 0.9, 0.2, 1.0])
                
                # Occasionally place ramps (every 7th)
                if i % 7 == 6:
                    self._create_ramp([x_pos + spacing, start_position[1], 0.0], size=[0.3, 0.4, 0.08], angle=12, color=[0.6, 0.6, 0.6, 1.0])
                    
            return self.obstacles
            
        elif course_type == 'random':
            # Random obstacle course
            obstacle_types = ['box', 'sphere', 'cylinder', 'ramp']
            num_obstacles = int(length / 0.3)  # Place obstacles at approximately 0.3m intervals
            
            np.random.seed(None)  # Ensure randomness
            
            for i in range(num_obstacles):
                obs_type = obstacle_types[np.random.randint(0, len(obstacle_types))]
                # Start from 0.1m (place close)
                x_pos = start_position[0] + 0.1 + i * 0.3 + np.random.uniform(0.05, 0.15)
                y_pos = start_position[1] + np.random.uniform(-0.25, 0.25)
                
                if obs_type == 'box':
                    size = [np.random.uniform(0.05, 0.12), np.random.uniform(0.05, 0.12), np.random.uniform(0.05, 0.1)]
                    self._create_box([x_pos, y_pos, size[2]/2], size)
                    
                elif obs_type == 'sphere':
                    radius = np.random.uniform(0.04, 0.08)
                    self._create_sphere([x_pos, y_pos, radius], radius)
                    
                elif obs_type == 'cylinder':
                    radius = np.random.uniform(0.04, 0.08)
                    height = np.random.uniform(0.08, 0.16)
                    self._create_cylinder([x_pos, y_pos, height/2], radius, height)
                    
                elif obs_type == 'ramp':
                    size = [np.random.uniform(0.2, 0.4), np.random.uniform(0.2, 0.3), np.random.uniform(0.05, 0.08)]
                    angle = np.random.uniform(5, 15)
                    self._create_ramp([x_pos, y_pos, 0.0], size, angle)
            
            return self.obstacles
        
        else:
            raise ValueError(f"Unknown course type: {course_type}. Available types: 'simple', 'dense', 'random'")
    
    def remove_all_obstacles(self):
        """Remove all obstacles"""
        for obstacle in self.obstacles:
            if isinstance(obstacle['id'], list):
                for obs_id in obstacle['id']:
                    p.removeBody(obs_id)
            else:
                p.removeBody(obstacle['id'])
        
        self.obstacles = []
    
    def get_obstacle_positions(self):
        """
        Get position information for all obstacles
        
        Returns:
            list: List of dictionaries containing information for each obstacle (ID, type, position)
        """
        positions = []
        for obstacle in self.obstacles:
            if isinstance(obstacle['id'], list):
                for obs_id in obstacle['id']:
                    pos, _ = p.getBasePositionAndOrientation(obs_id)
                    positions.append({
                        'id': obs_id,
                        'type': obstacle['type'],
                        'position': pos
                    })
            else:
                pos, _ = p.getBasePositionAndOrientation(obstacle['id'])
                positions.append({
                    'id': obstacle['id'],
                    'type': obstacle['type'],
                    'position': pos
                })
        
        return positions