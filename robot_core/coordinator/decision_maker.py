from enum import Enum
from transitions import Machine
from dataclasses import dataclass
import time

from robot_core.utils.position import Position, PositionTypes
from robot_core.orchestration.scan_point_utils import ScanPoint, ScanPointGenerator
from robot_core.perception.detection_results import BallDetection, BoxDetection, DetectionResult
from robot_core.hardware.dimensions import COURT_XLIM, COURT_YLIM
from robot_core.coordinator.commands import RobotCommands, StateWrapper, VisionCommands
from robot_core.utils.command_utils import StatefulCommandQueue, Command, CommandStatus
import enum

class RobotStates(Enum):
    IDLE = 0 # Doing nothing
    DRIVE_TO_BALL = 1 # implemented
    DRIVE_TO_DEPOSIT = 2
    DRIVE_TO_SCAN_POINT = 3 # Exploring the court.
    ROTATE_SCAN = 4

    STAMP = 5
    ALIGN = 6
    DEPOSIT = 7

class DecisionMaker:
    def __init__(
            self,
            robot_pose,
            goal_position,
            detection_results_q,
            shared_data,
            occupancy_map, # shared object instantiated inside ProcessCoordinator
            deposit_time_limit=8*60,
            max_capacity=4,
            collection_box_location=(0, 0)
    ):
        # Shared multiprocessing manager variables
        self.robot_pose = robot_pose  # Shared dict with robot's current pose {'x': x, 'y': y, 'th': th}
        self.goal_position = goal_position  # Shared dict for the goal position
        self.detection_results_q = detection_results_q  # Shared queue with the latest detection results
        self.shared_data = shared_data # Shared dict with the latest detection results
        self.command_queue = shared_data['command_queue']  # Shared command queue

        # Shared ProcessCoordinator variables (these are plain old lists, dicts, not mp.manager objects)
        self.occupancy_map = occupancy_map  # Shared list for the occupancy map

        # Parameters
        self.max_capacity = max_capacity
        self.deposit_time_limit = deposit_time_limit  # Example value: 5 minutes
        self.collection_box_location = collection_box_location

        # Other things to keep track of state
        self.collected_balls = 0
        self.last_deposit_time = time.time()
        self.current_ball_stamp_attempts = 0
        self.current_ball_position = None


        # Define states and transitions
        states = [

            {'name': RobotStates.IDLE},
            {'name': RobotStates.DRIVE_TO_BALL, 'on_enter': 'set_goal_to_nearest_ball'}, # need to issue RobotCommands.DRIVE command
            {'name': RobotStates.DRIVE_TO_DEPOSIT, 'on_enter': 'set_goal_to_deposit_box'}, # need to issue RobotCommands.DRIVE command
            {'name': RobotStates.DRIVE_TO_SCAN_POINT, 'on_enter': 'set_goal_to_scan_point'}, # need to issue RobotCommands.DRIVE command
            {'name': RobotStates.ROTATE_SCAN}, # need to issue RobotCommands.ROTATE command
            {'name': RobotStates.STAMP, 'on_enter': 'start_stamping', 'on_exit': 'handle_stamp_completion'},  # need to issue RobotCommands.ROTATE command
            {'name': RobotStates.ALIGN, 'on_enter': 'start_aligning', 'on_exit': ''},  # need to issue RobotCommands.ALIGN command
            {'name': RobotStates.DEPOSIT, 'on_enter': 'start_depositing', 'on_exit': 'handle_deposit_completion'},  # need to issue RobotCommands.DEPOSIT command

        ]

        transitions = [
            ##### These are the transitions that determine the robot's next action based on its internal state ####
            # Transitions for driving to a location
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.DRIVE_TO_BALL, RobotStates.DRIVE_TO_SCAN_POINT, RobotStates.DEPOSIT, RobotStates.STAMP, RobotStates.IDLE],
                'dest': RobotStates.DRIVE_TO_BALL,
                'conditions': ['has_known_balls'], # needs to return True
                'unless': ['should_return_to_deposit'] # needs to return False
            },
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.DRIVE_TO_DEPOSIT, RobotStates.DRIVE_TO_BALL, RobotStates.DRIVE_TO_SCAN_POINT, RobotStates.STAMP], # i.e. NOT DEPOSIT, ALIGN
                'dest': RobotStates.DRIVE_TO_DEPOSIT,
                'conditions': ['should_return_to_deposit'],
            },
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.DRIVE_TO_BALL, RobotStates.DRIVE_TO_SCAN_POINT, RobotStates.DEPOSIT, RobotStates.STAMP, RobotStates.IDLE],
                'dest': RobotStates.DRIVE_TO_SCAN_POINT,
                'conditions': [],
                'unless': ['has_known_balls', 'should_return_to_deposit']
            },

            # Transitions for performing an action (stamp, align, deposit)
            {
                'trigger': 'decide_next_action',
                'source': RobotStates.DRIVE_TO_BALL,
                'dest': RobotStates.STAMP,
                'conditions': ['is_ball_in_front'],
                'unless': ['should_return_to_deposit']
            },
            {
                'trigger': 'decide_next_action',
                'source': RobotStates.DRIVE_TO_DEPOSIT,
                'dest': RobotStates.ALIGN,
                'conditions': ['should_return_to_deposit','is_at_deposit_box'],
            },
            {
                'trigger': 'decide_next_action',
                'source': RobotStates.ALIGN,
                'dest': RobotStates.DEPOSIT,
            },

        ]

        # Initialize the state machine
        self.machine = Machine(model=self, states=states, transitions=transitions, initial=RobotStates.IDLE, after_state_change='after_state_change', before_state_change='before_state_change')


    def update(self):
        """Main update called in ProcessCoordinator control loop."""
        print(f'Updating. Current state: {self.state}')
        self.update_sensor_data()
        self.check_state_completion()


    def after_state_change(self):
        # issue the command to the command queue
        print(f"Current state: {self.state}")

    def check_state_completion(self):
        ### Actions to take when a state is completed, or still in progress.
        if self.state == RobotStates.DRIVE_TO_BALL and self._is_drive_done():
            self.handle_drive_completion()
        elif self.state == RobotStates.STAMP and self._is_stamp_done():
            self.handle_stamp_completion()
        elif self.state == RobotStates.ALIGN and self._is_align_done():
            self.handle_align_completion()
        elif self.state == RobotStates.DEPOSIT and self._is_deposit_done():
            self.handle_deposit_completion()
        elif self.state == RobotStates.SEARCH and self._is_search_done():
            self.handle_search_completion()

    def update_sensor_data(self):
        # Implement logic to handle external events or commands
        # Don't need to pass perception data to occupancy map. That's done in ProcessCoordinator.
        pass

    def handle_drive_completion(self):
        if self.goal_position.position_type == PositionTypes.BALL:
            self.stamp()
        elif self.goal_position.position_type == PositionTypes.BOX:
            self.align()
        else:
            self.next_action()

    def handle_stamp_completion(self):
        # If stamping is successful, increment the collected balls, AND remove from the occupancy map
        self.collected_balls += 1
        self.occupancy_map.remove_ball(self.current_ball_position)


    # After depositing, reset the collected balls and update the last deposit time
    def handle_deposit_completion(self):
        self.collected_balls = 0
        self.last_deposit_time = time.time()



    # Goal setting functions
    def set_goal_to_scan_point(self):
        self.goal_position.update({
            'goal': ScanPointGenerator.get_next_scan_point(self.curr_scan_point, self.prev_scan_point),
            'time': time.time()
        })

    def set_goal_to_nearest_ball(self):
        nearest_ball = self.occupancy_map.get_nearest_ball(self.robot_pose['x'], self.robot_pose['y'], self.robot_pose['th'])
        self.current_ball_position = nearest_ball
        self.goal_position.update({
            'goal': Position(*nearest_ball, 0, PositionTypes.BALL),
            'time': time.time()
        })

    def set_goal_to_deposit_box(self):
        self.goal_position.update({
            'goal': Position(*self.collection_box_location, 0, PositionTypes.BOX),
            'time': time.time()
        })

    ####### State checks, truth conditions #######

    # Returns True if occupancy map is not empty
    def has_known_balls(self):
        return not self.occupancy_map.is_empty()

    # Returns True if robot should return to deposit box. Depends on time elapsed and capacity.
    def should_return_to_deposit(self):
        time_condition = time.time() - self.last_deposit_time > self.deposit_time_limit
        capacity_condition = self.collected_balls >= self.max_capacity and self.collected_balls > 0

        if time_condition or capacity_condition: return True
        return False

    def is_at_deposit_box(self):
        # insert logic to check if robot is at deposit box. For now, just return True
        pass

    def is_ball_in_front(self):
        # insert logic to check if ball is in front. For now, just return True
        pass

# Example usage
robot = Robot()
while True:
    robot.update()
    time.sleep(2)  # Add a small delay to prevent busy-waiting