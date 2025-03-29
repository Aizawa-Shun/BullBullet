import pybullet as p
import pybullet_data
from time import sleep
import locale
from env.lidar_sensor import LidarSensor
from env.obstacle_generator import ObstacleGenerator
from env.goal_marker import GoalMarker

class RobotEnvironment:
    """Simulation environment class for quadruped robots"""
    
    def __init__(self, urdf_path, robot_pos=[0, 0, 0.08], robot_rot=[0, 0, 0], use_gui=True, cam=True):
        """
        Initialize the simulation environment
        
        Args:
            urdf_path: Path to the robot's URDF file
            robot_pos: Initial position of the robot
            robot_rot: Initial orientation of the robot (Euler angles)
            use_gui: Whether to use GUI
            cam: Whether to make the camera follow the robot
        """
        self.urdf_path = urdf_path
        self.robot_pos = robot_pos
        self.robot_rot = p.getQuaternionFromEuler(robot_rot)
        self.use_gui = use_gui
        self.cam = cam
        
        # Initialize simulation
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1/240)
        
        # Load ground and robot
        locale.setlocale(locale.LC_ALL, "ja_JP.UTF-8")
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = self._load_robot()
        
        # Store joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_info = self._get_joint_info()
        
        # Initialize obstacle generator
        self.obstacle_generator = ObstacleGenerator(self)
        self.has_obstacles = False
        
        # Initialize goal-related attributes
        self.goal_marker = None
        self.has_goal = False
        self.goal_reached = False
        
    def _load_robot(self):
        """Load the robot's URDF and return its ID"""
        robot_id = p.loadURDF(self.urdf_path, self.robot_pos, self.robot_rot)
        return robot_id
    
    def _get_joint_info(self):
        """Collect information for all robot joints"""
        joint_info = {}
        for joint_index in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, joint_index)
            joint_info[joint_index] = {
                'name': info[1].decode('utf-8'),
                'type': info[2],
                'lower_limit': info[8],
                'upper_limit': info[9],
                'max_force': info[10],
                'max_velocity': info[11]
            }
        return joint_info
    
    def set_physics_parameters(self, gravity, timestep):
        """Set physics parameters"""
        # Set gravity
        p.setGravity(*gravity)
        
        # Set timestep
        p.setTimeStep(timestep)
    
    def apply_action(self, action, max_force=5.0):
        """Apply joint actions to the robot"""
        for joint_index, target_pos in enumerate(action):
            if joint_index < self.num_joints:
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=max_force
                )
    
    def get_robot_state(self):
        """Return the current state of the robot"""
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot_id)
        
        # Get base velocity information
        linear_vel, angular_vel = p.getBaseVelocity(self.robot_id)
        
        joint_states = [p.getJointState(self.robot_id, i) for i in range(self.num_joints)]
        return {
            'base_position': base_pos,
            'base_orientation': base_orient,
            'base_linear_velocity': linear_vel,  # Add linear velocity
            'base_angular_velocity': angular_vel,  # Add angular velocity
            'joint_states': joint_states
        }
    
    def update_camera(self):
        """Make the camera follow the robot"""
        # Get robot position
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        # Camera parameters
        distance = 0.5
        yaw = 45.0
        pitch = -30.0
        
        # Set robot position as the focus point
        target_pos = [
            robot_pos[0],  # x
            robot_pos[1],  # y
            robot_pos[2]   # z
        ]
        
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target_pos
        )
    
    def step(self):
        """Advance the simulation by one step"""
        p.stepSimulation()
        if self.use_gui and self.cam: 
            self.update_camera()
        if self.use_gui:
            sleep(1/240)
            
    def add_lidar(self, lidar_link_index=None, num_rays=36, ray_length=3.0, ray_start_length=0.1):
        """
        Add a LIDAR sensor to the robot
        
        Args:
            lidar_link_index: Index of the link where the LIDAR sensor is attached (Base link if None)
            num_rays: Number of rays
            ray_length: Maximum length of rays
            ray_start_length: Starting distance of rays
        
        Returns:
            LidarSensor: Instance of the created LIDAR sensor
        """
        self.lidar_sensor = LidarSensor(
            robot_id=self.robot_id,
            lidar_link_index=lidar_link_index,
            num_rays=num_rays,
            ray_length=ray_length,
            ray_start_length=ray_start_length
        )
        
        self.has_lidar = True
        return self.lidar_sensor

    def remove_lidar(self):
        """Remove the LIDAR sensor"""
        if hasattr(self, 'lidar_sensor') and self.lidar_sensor is not None:
            self.lidar_sensor.update_visualization(show=False)
            self.lidar_sensor = None
            self.has_lidar = False

    def scan_environment(self):
        """
        Scan the environment using the LIDAR sensor
        
        Returns:
            dict: Scan results
        """
        if not hasattr(self, 'lidar_sensor') or self.lidar_sensor is None:
            raise ValueError("LIDAR sensor is not set. Call the add_lidar method first.")
        
        return self.lidar_sensor.scan()

    def get_robot_state_with_lidar(self):
        """
        Get robot state and LIDAR scan results
        
        Returns:
            dict: Dictionary containing robot state and LIDAR scan results
        """
        robot_state = self.get_robot_state()
        
        if hasattr(self, 'lidar_sensor') and self.lidar_sensor is not None:
            scan_result = self.lidar_sensor.scan()
            robot_state['lidar'] = scan_result
            
            # Additional analysis results
            sectors = self.lidar_sensor.get_sector_analysis(scan_result)
            closest_dir, closest_dist, _, _ = self.lidar_sensor.get_closest_obstacle_direction(scan_result)
            
            robot_state['lidar']['sectors'] = sectors
            robot_state['lidar']['closest_obstacle'] = {
                'direction': closest_dir,  # -1.0 (left) to 1.0 (right)
                'distance': closest_dist
            }
        
        # Add obstacle information
        if hasattr(self, 'has_obstacles') and self.has_obstacles:
            obstacle_positions = self.obstacle_generator.get_obstacle_positions()
            robot_state['obstacles'] = obstacle_positions
        
        # Add goal information
        if hasattr(self, 'goal_marker') and self.goal_marker is not None:
            robot_pos = robot_state['base_position']
            robot_state['goal'] = {
                'position': self.goal_marker.position,
                'radius': self.goal_marker.radius,
                'reached': getattr(self, 'goal_reached', False)
            }
            
            # Calculate distance to goal
            dx = self.goal_marker.position[0] - robot_pos[0]
            dy = self.goal_marker.position[1] - robot_pos[1]
            distance = (dx**2 + dy**2)**0.5
            robot_state['goal']['distance'] = distance
        
        return robot_state
        
    def get_full_state(self):
        """
        Get the complete state of the robot (including LIDAR, obstacles, and goal information)
        
        Returns:
            dict: Complete state of the robot
        """
        return self.get_robot_state_with_lidar()
    
    def add_obstacles(self, course_type='simple', start_position=None, length=5.0):
        """
        Add an obstacle course
        
        Args:
            course_type: Course type ('simple', 'dense', 'random')
            start_position: Starting position (In front of the robot if None)
            length: Length of the course
            
        Returns:
            list: List of added obstacles
        """
        if start_position is None:
            # Place obstacle course near the front of the robot (closer)
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            start_position = [robot_pos[0] + 0.1, robot_pos[1], 0.0]  # Start 0.1m ahead
        
        obstacles = self.obstacle_generator.create_obstacle_course(course_type, start_position, length)
        self.has_obstacles = True
        return obstacles
    
    def clear_obstacles(self):
        """Remove all obstacles"""
        self.obstacle_generator.remove_all_obstacles()
        self.has_obstacles = False
    
    def get_robot_state_with_obstacles(self):
        """Get robot state and surrounding obstacle information"""
        robot_state = self.get_robot_state()
        if self.has_obstacles:
            obstacle_positions = self.obstacle_generator.get_obstacle_positions()
            robot_state['obstacles'] = obstacle_positions
        
        return robot_state
    
    def add_goal(self, goal_position=None, radius=0.3, color=None):
        """
        Add a goal marker to the simulation environment
        
        Args:
            goal_position (list, optional): Goal position. Auto-placed in front if None
            radius (float): Goal radius
            color (list, optional): Goal color. Default color if None
        
        Returns:
            GoalMarker: Instance of the created goal marker
        """
        if goal_position is None:
            # Place the goal at a suitable position in front of the robot
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            goal_position = [robot_pos[0] + 3.0, robot_pos[1], 0.0]
        
        if color is None:
            # Default semi-transparent green
            color = [0.0, 0.8, 0.0, 0.5]
        
        # Create goal marker
        self.goal_marker = GoalMarker(position=goal_position, radius=radius, color=color)
        self.goal_marker.create()
        self.goal_marker.add_visual_ring()
        
        # Goal-related state information
        self.has_goal = True
        self.goal_reached = False
        
        return self.goal_marker

    def remove_goal(self):
        """Remove the goal marker from the simulation environment"""
        if hasattr(self, 'goal_marker') and self.goal_marker is not None:
            self.goal_marker.remove()
            self.goal_marker = None
            self.has_goal = False
            self.goal_reached = False

    def update_goal_position(self, new_position):
        """Update the goal marker position"""
        if hasattr(self, 'goal_marker') and self.goal_marker is not None:
            self.goal_marker.update_position(new_position)
            self.goal_reached = False

    def check_goal_reached(self):
        """Check if the robot has reached the goal"""
        if not hasattr(self, 'goal_marker') or self.goal_marker is None:
            return False
        
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        reached = self.goal_marker.check_reached(robot_pos)
        
        # Handle first-time goal reaching
        if reached and not self.goal_reached:
            self.goal_reached = True
            print("🎉 Goal reached! 🎉")
            
            # Goal reached effect (optional)
            self._show_goal_reached_effect()
        
        return reached

    def _show_goal_reached_effect(self):
        """Display visual effect when goal is reached"""
        if not self.use_gui:
            return
        
        # Get goal position
        goal_pos = self.goal_marker.position
        
        # Display text
        p.addUserDebugText(
            text="🏆 GOAL! 🏆",
            textPosition=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.3],
            textColorRGB=[1, 0.8, 0],
            textSize=2.0,
            lifeTime=3.0
        )

    def get_robot_state_with_goal(self):
        """Get robot state and goal information"""
        robot_state = self.get_robot_state()
        
        if hasattr(self, 'goal_marker') and self.goal_marker is not None:
            robot_state['goal'] = {
                'position': self.goal_marker.position,
                'radius': self.goal_marker.radius,
                'reached': self.goal_reached
            }
            
            # Calculate distance to goal
            robot_pos = robot_state['base_position']
            dx = self.goal_marker.position[0] - robot_pos[0]
            dy = self.goal_marker.position[1] - robot_pos[1]
            distance = (dx**2 + dy**2)**0.5
            robot_state['goal']['distance'] = distance
        
        return robot_state
    
    def close(self):
        """Close the PyBullet connection"""
        p.disconnect(self.physics_client)