import pybullet as p
import numpy as np

class GoalMarker:
    """Semi-transparent circular marker indicating the robot's goal position"""
    
    def __init__(self, position=[1.0, 0.0, 0.0], radius=0.3, height=0.2, color=[0.0, 0.8, 0.0, 0.5]):
        """
        Initialize a semi-transparent circular goal marker
        
        Args:
            position (list): Goal position [x, y, z]
            radius (float): Goal radius
            height (float): Goal height
            color (list): Goal color [r, g, b, a] (a is transparency)
        """
        self.position = position
        self.radius = radius
        self.height = height
        self.color = color
        self.goal_id = None
        self.debug_items = []
        
        # Distance threshold for goal achievement
        self.reach_threshold = radius * 0.7
    
    def create(self):
        """Create the goal marker"""
        # Create only visual shape of cylinder (no collision shape)
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            length=self.height,
            rgbaColor=self.color
        )
        
        # Create as a non-colliding object with mass 0
        self.goal_id = p.createMultiBody(
            baseMass=0,  # Mass 0 makes it immovable
            baseCollisionShapeIndex=-1,  # No collision shape
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[self.position[0], self.position[1], self.position[2] + self.height/2]  # Adjust height to be on the ground
        )
        
        return self.goal_id
    
    def add_visual_ring(self, segments=32):
        """Add a ring visual effect around the goal"""
        center_x, center_y = self.position[0], self.position[1]
        z_pos = self.position[2] + 0.01  # Slightly above the ground
        
        # Generate points for the ring
        points = []
        for i in range(segments + 1):
            angle = 2.0 * np.pi * i / segments
            x = center_x + self.radius * np.cos(angle)
            y = center_y + self.radius * np.sin(angle)
            points.append([x, y, z_pos])
        
        # Draw the ring with debug lines
        line_color = self.color[:3]  # RGB only
        
        for i in range(segments):
            item_id = p.addUserDebugLine(
                lineFromXYZ=points[i],
                lineToXYZ=points[i+1],
                lineColorRGB=line_color,
                lineWidth=3.0
            )
            self.debug_items.append(item_id)
        
        # Add a marker in the center
        center_marker = p.addUserDebugText(
            text="GOAL",
            textPosition=[center_x, center_y, z_pos + 0.05],
            textColorRGB=[1, 1, 1],
            textSize=1.5
        )
        self.debug_items.append(center_marker)
        
        # Add an arrow
        arrow_length = self.radius * 1.5
        arrow_start = [center_x, center_y, z_pos + 0.3]
        arrow_end = [center_x, center_y, z_pos + 0.01]
        
        arrow = p.addUserDebugLine(
            lineFromXYZ=arrow_start,
            lineToXYZ=arrow_end,
            lineColorRGB=[1, 1, 0],
            lineWidth=2.0
        )
        self.debug_items.append(arrow)
    
    def check_reached(self, robot_position):
        """Check if the robot has reached the goal"""
        distance_xy = np.sqrt(
            (robot_position[0] - self.position[0])**2 + 
            (robot_position[1] - self.position[1])**2
        )
        return distance_xy < self.reach_threshold
    
    def update_position(self, new_position):
        """Update the goal position"""
        self.position = new_position
        
        # Remove existing marker
        if self.goal_id is not None:
            p.removeBody(self.goal_id)
        
        for item in self.debug_items:
            p.removeUserDebugItem(item)
        self.debug_items = []
        
        # Create marker at new position
        self.create()
        self.add_visual_ring()
        
    def remove(self):
        """Remove the goal marker"""
        if self.goal_id is not None:
            p.removeBody(self.goal_id)
            self.goal_id = None
        
        for item in self.debug_items:
            p.removeUserDebugItem(item)
        self.debug_items = []