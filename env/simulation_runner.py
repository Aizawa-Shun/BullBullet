import os
import time
from time import sleep
from env.quad_env import RobotEnvironment
from env.gait import GaitGenerator
from env.config_loader import ConfigLoader
import logging
from utils.logger import LoggerManager

class SimulationRunner:
    """Class for managing quadruped robot simulation execution"""
    
    def __init__(self, config_file=None, logger_manager=None, console_output=False):
        """
        Initialize the simulation runner
        
        Args:
            config_file (str, optional): Path to configuration file. Default settings used if None.
            logger_manager (LoggerManager, optional): Logger manager. Creates a new one if None.
            console_output (bool): Whether to output logs to console as well
        """
        # Logger setup
        if logger_manager is None:
            self.logger_manager = LoggerManager()
        else:
            self.logger_manager = logger_manager
        
        self.logger = self.logger_manager.get_logger('simulation', console_output=console_output)
        self.main_logger = self.logger_manager.get_logger('main')
        self.console_output = console_output
        
        # Load configuration
        self.config = ConfigLoader(config_file)
        self.config_file = config_file
        
        # Robot environment instance
        self.env = None
        
        # Gait generator instance
        self.gait_generator = None
        
        # Simulation related flags
        self.lidar_enabled = False
        self.goal_enabled = False
        
        self.log_info(f"SimulationRunner initialization complete - Configuration file: {config_file}")
    
    def log_info(self, message):
        """Output information log (both to file and console)"""
        self.logger.info(message)
        if self.console_output:
            print(message)
    
    def setup_environment(self):
        """Set up the environment"""
        # Get robot environment settings
        robot_config = self.config.get('robot')
        env_config = self.config.get('environment')
        
        # Initialize environment
        urdf_path = robot_config['urdf_path']
        robot_pos = robot_config['position']
        robot_rot = robot_config['rotation']
        max_force = robot_config['max_force']
        
        self.log_info(f"Initializing environment: URDF={urdf_path}, pos={robot_pos}, rot={robot_rot}")
        
        # Initialize environment
        self.env = RobotEnvironment(
            urdf_path=urdf_path, 
            robot_pos=robot_pos, 
            robot_rot=robot_rot, 
            use_gui=env_config['use_gui'], 
            cam=env_config['camera_follow']
        )
        
        # Set physics parameters for the environment
        import pybullet as p
        p.setGravity(*env_config['gravity'])
        p.setTimeStep(env_config['timestep'])
        
        self.log_info(f"Physics parameters set: gravity={env_config['gravity']}, timestep={env_config['timestep']}")
        
        return self.env
    
    def setup_gait_generator(self):
        """Set up gait generator"""
        gait_config = self.config.get('gait')
        
        self.gait_generator = GaitGenerator(
            amplitude=gait_config['amplitude'],
            frequency=gait_config['frequency']
        )
        self.gait_generator.set_turn_direction(gait_config['turn_direction'])
        self.gait_generator.set_turn_intensity(gait_config['turn_intensity'])
        self.gait_generator.set_backward(False)  # Forward movement by default
        self.gait_generator.set_gait_pattern(gait_config['pattern'])
        
        self.log_info(f"Gait generator configured: pattern={gait_config['pattern']}, amplitude={gait_config['amplitude']}, frequency={gait_config['frequency']}")
        
        return self.gait_generator
    
    def setup_sensors_and_environment(self):
        """Set up sensors and environment objects"""
        # LiDAR sensor setup
        lidar_config = self.config.get('lidar')
        self.lidar_enabled = lidar_config['enabled']
        
        if self.lidar_enabled:
            lidar = self.env.add_lidar(
                num_rays=lidar_config['num_rays'],
                ray_length=lidar_config['ray_length'],
                ray_start_length=lidar_config['ray_start_length']
            )
            self.log_info(f"LiDAR sensor initialized: rays={lidar_config['num_rays']}, length={lidar_config['ray_length']}")
        
        # Obstacle setup
        obstacle_config = self.config.get('obstacles')
        if obstacle_config['enabled']:
            obstacles = self.env.add_obstacles(
                course_type=obstacle_config['course_type'],
                length=obstacle_config['length']
            )
            self.log_info(f"Obstacle course initialized: type={obstacle_config['course_type']}, length={obstacle_config['length']}")
        
        # Goal setup
        goal_config = self.config.get('goal')
        self.goal_enabled = goal_config['enabled']
        
        if self.goal_enabled:
            goal = self.env.add_goal(
                goal_position=goal_config['position'],
                radius=goal_config['radius'],
                color=goal_config['color'] if 'color' in goal_config else None
            )
            self.log_info(f"Goal initialized: position={goal_config['position']}, radius={goal_config['radius']}")
    
    def run_simulation(self):
        """Execute simulation loop"""
        # Simulation settings
        sim_config = self.config.get('simulation')
        max_steps = sim_config['max_steps']
        debug_interval = sim_config['debug_interval']
        max_force = self.config.get('robot')['max_force']
        
        print(f"Starting robot walking simulation - Max steps: {max_steps}")
        self.log_info(f"Starting robot walking simulation - Max steps: {max_steps}")
        
        # Initial LiDAR scan
        if self.lidar_enabled:
            scan_result = self.env.scan_environment()
            self.log_info(f"Initial scan complete - Detected {scan_result['num_hits']} obstacles")
        
        try:
            start_time = time.time()
            goal_reached = False
            
            for step in range(max_steps):
                # Get action from gait generator
                action = self.gait_generator.get_action()
                
                # Apply action to robot
                self.env.apply_action(action, max_force)
                
                # Advance simulation by one step
                self.env.step()
                
                # Check if goal reached
                if self.goal_enabled and self.env.check_goal_reached():
                    goal_reached = True
                    elapsed_time = time.time() - start_time
                    print(f"Goal reached at step {step}! Elapsed time: {elapsed_time:.2f} seconds")
                    self.log_info(f"Goal reached at step {step}! Elapsed time: {elapsed_time:.2f} seconds")
                    # Wait a bit to celebrate reaching the goal
                    sleep(2)
                    break
                
                # Execute LiDAR scan
                if self.lidar_enabled and step % 10 == 0:  # Scan every 10 steps
                    scan_result = self.env.scan_environment()
                
                # Display status periodically
                if step % debug_interval == 0:
                    state = self.env.get_robot_state()
                    elapsed_time = time.time() - start_time
                    status_msg = f"Step {step}: Position: {state['base_position']}, Elapsed time: {elapsed_time:.2f} seconds"
                    if self.console_output and step % (debug_interval * 5) == 0:  # Show to console less frequently
                        print(status_msg)
                    self.logger.info(status_msg)
            
            # End of simulation
            total_time = time.time() - start_time
            
            if goal_reached:
                result_msg = f"Simulation successful: Goal reached - Total time: {total_time:.2f} seconds, Steps: {step+1}"
            else:
                result_msg = f"Simulation complete: Time limit reached - Total time: {total_time:.2f} seconds, Steps: {max_steps}"
            
            print(result_msg)
            self.log_info(result_msg)
        
        except Exception as e:
            error_msg = f"Error occurred during simulation: {e}"
            print(error_msg)
            self.logger.error(error_msg, exc_info=True)
        
        finally:
            self.log_info("Simulation completed")
    
    def run(self):
        """Start the simulation execution"""
        try:
            # Set up environment
            self.setup_environment()
            
            # Set up gait generator
            self.setup_gait_generator()
            
            # Set up sensors and environment objects
            self.setup_sensors_and_environment()
            
            # Run simulation
            self.run_simulation()
            
        except Exception as e:
            error_msg = f"Error occurred during simulation: {e}"
            print(error_msg)
            self.logger.error(error_msg, exc_info=True)
            if self.env:
                self.env.close()
            raise
        finally:
            if self.env:
                self.env.close()
                self.log_info("Environment closed")