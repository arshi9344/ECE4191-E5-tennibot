from enum import Enum
from transitions import Machine
from dataclasses import dataclass
import time
import multiprocessing as mp
from typing import List, Tuple, Optional

from robot_core.utils.position import Position, PositionTypes
from robot_core.orchestration.scan_point_utils import ScanPoint, ScanPointGenerator
from robot_core.perception.detection_results import BallDetection, BoxDetection
from robot_core.hardware.dimensions import COURT_XLIM, COURT_YLIM
from robot_core.coordinator.commands import RobotCommands, StateWrapper, VisionCommands
from robot_core.utils.command_utils import StatefulCommandQueue, Command, CommandStatus
from robot_core.coordinator.occupancy_map import OccupancyMap
import enum

class RobotStates(Enum):
    IDLE = 0 # Doing nothing
    DRIVE_TO_BALL = 1 # implemented
    DRIVE_TO_DEPOSIT_BOX = 2
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
            command_queue, # the shared command queue inside shared_data
            occupancy_map, # shared object instantiated inside ProcessCoordinator
            deposit_time_limit=8*60,
            max_capacity=4,
            collection_box_location=(COURT_XLIM, COURT_YLIM)
    ):
        # Shared multiprocessing manager variables
        self.robot_pose = robot_pose  # Shared dict with robot's current pose {'x': x, 'y': y, 'th': th}
        self.goal_position = goal_position  # Shared dict for the goal position
        self.detection_results_q = detection_results_q  # Shared queue with the latest detection results
        self.shared_data = shared_data # Shared dict with the latest detection results
        self.command_queue : StatefulCommandQueue = command_queue # Shared command queue

        # Shared ProcessCoordinator variables (these are plain old lists, dicts, not mp.manager objects)
        self.occupancy_map : OccupancyMap = occupancy_map  # Shared occupancy map object

        # Parameters
        self.max_capacity = max_capacity
        self.deposit_time_limit = deposit_time_limit  # Example value: 5 minutes
        self.collection_box_location : Tuple[float, float] = collection_box_location

        # Other things to keep track of state
        self.collected_balls = 0
        self.last_deposit_time = time.time()
        self.current_ball_stamp_attempts = 0
        self.current_ball_id = None
        self.scan_point_gen = ScanPointGenerator()
        self.curr_command_id = None

        # Define states and transitions
        states = [
            {'name': RobotStates.IDLE},
            {'name': RobotStates.DRIVE_TO_BALL, 'on_enter': 'set_goal_to_nearest_ball'}, # need to issue RobotCommands.DRIVE command
            {'name': RobotStates.DRIVE_TO_DEPOSIT_BOX, 'on_enter': 'set_goal_to_deposit_box'}, # need to issue RobotCommands.DRIVE command
            {'name': RobotStates.DRIVE_TO_SCAN_POINT, 'on_enter': 'set_goal_to_scan_point'}, # need to issue RobotCommands.DRIVE command
            {'name': RobotStates.ROTATE_SCAN}, # need to issue RobotCommands.ROTATE command
            {'name': RobotStates.STAMP, 'on_enter': '', 'on_exit': 'handle_stamp_completion'},
            {'name': RobotStates.ALIGN, 'on_enter': '', 'on_exit': ''},  # need to issue RobotCommands.ALIGN command
            {'name': RobotStates.DEPOSIT, 'on_enter': '', 'on_exit': 'handle_deposit_completion'},  # need to issue RobotCommands.DEPOSIT command
        ]

        transitions = [
            ##### These are the transitions that determine the robot's next action based on its internal state ####

            # Transitions for driving to a location
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.IDLE, RobotStates.DRIVE_TO_SCAN_POINT, RobotStates.DEPOSIT, RobotStates.ROTATE_SCAN],
                'dest': RobotStates.DRIVE_TO_BALL,
                'conditions': ['has_known_balls'], # needs to return True
                'unless': ['should_return_to_deposit'] # needs to return False
            },
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.DRIVE_TO_BALL, RobotStates.DRIVE_TO_DEPOSIT_BOX, RobotStates.DRIVE_TO_SCAN_POINT, RobotStates.STAMP, RobotStates.ROTATE_SCAN], # i.e. NOT IDLE, DEPOSIT, ALIGN
                'dest': RobotStates.DRIVE_TO_DEPOSIT_BOX,
                'conditions': ['should_return_to_deposit']
            },
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.IDLE, RobotStates.ROTATE_SCAN, RobotStates.DEPOSIT], # could add RobotStates.STAMP, but permitted only if another ball is within 50cm or something
                'dest': RobotStates.DRIVE_TO_SCAN_POINT,
                'conditions': [],
                'unless': ['has_known_balls', 'should_return_to_deposit']
            },

            # Transitions for performing an action (stamp, align, deposit, rotate_scan)
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.DRIVE_TO_BALL, RobotStates.STAMP],
                'dest': RobotStates.STAMP,
                'conditions': ['is_ball_in_front'],
                'unless': ['should_return_to_deposit']
            },
            {
                'trigger': 'decide_next_action',
                'source': RobotStates.DRIVE_TO_DEPOSIT_BOX,
                'dest': RobotStates.ALIGN,
                'conditions': ['should_return_to_deposit','is_deposit_box_reached'],
            },
            {
                'trigger': 'decide_next_action',
                'source': RobotStates.ALIGN,
                'dest': RobotStates.DEPOSIT,
            },
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.DRIVE_TO_SCAN_POINT, RobotStates.STAMP],
                'dest': RobotStates.ROTATE_SCAN,
                'conditions': [],
            }

        ]

        # Initialize the state machine
        self.machine = Machine(model=self, states=states, transitions=transitions, initial=RobotStates.IDLE, after_state_change='_issue_command')


    def update(self):
        """Main update called in ProcessCoordinator control loop."""
        print(f'Updating. Current state: {self.state}')
        self._refresh_ball_goal() # this is to update the ball goal position to the nearest ball using most recent perception data
        self._check_command_completion()
        self.decide_next_action() # method added by the state machine at runtime

        #TODO: Refine state completion check logic here


    def _issue_command(self):
        # issue the command to the command queue
        match self.state:
            case RobotStates.DRIVE_TO_BALL:
                self.curr_command_id = self.command_queue.put(RobotCommands.DRIVE)
            case RobotStates.DRIVE_TO_DEPOSIT_BOX:
                self.curr_command_id = self.command_queue.put(RobotCommands.DRIVE)
            case RobotStates.DRIVE_TO_SCAN_POINT:
                self.curr_command_id = self.command_queue.put(RobotCommands.DRIVE)
            case RobotStates.ROTATE_SCAN:
                self.curr_command_id = self.command_queue.put(RobotCommands.ROTATE)
            case RobotStates.STAMP:
                self.curr_command_id = self.command_queue.put(RobotCommands.STAMP)
            case RobotStates.ALIGN:
                self.curr_command_id = self.command_queue.put(RobotCommands.ALIGN)
            case RobotStates.DEPOSIT:
                self.curr_command_id = self.command_queue.put(RobotCommands.DEPOSIT)
            case RobotStates.IDLE:
                self.curr_command_id = self.command_queue.put(RobotCommands.STOP)

        print(f"Current state: {self.state}, command_id: {self.curr_command_id}")


    # Check if the last issued A.K.A current command is completed or still processing.
    def _check_command_completion(self) -> bool:
        return_val = False
        if self.curr_command_id is not None:
            status = self.command_queue.get_status(self.curr_command_id)
            if status == CommandStatus.DONE:
                self.curr_command_id = None
                return_val = True
            elif status == CommandStatus.FAILED:
                self.curr_command_id = None
                return_val = True
            elif status == CommandStatus.PROCESSING:
                return_val = False
            elif status == CommandStatus.QUEUED:
                return_val = False
            print(f"{self.command_queue.get_data(self.curr_command_id)} (id: {self.curr_command_id}) is {status}")
        else:
            print(f"No current command")

        return return_val

    def _refresh_ball_goal(self):
        if self.state == RobotStates.DRIVE_TO_BALL:
            self.set_goal_to_nearest_ball()

    # def handle_drive_completion(self):
    #     if self.goal_position.position_type == PositionTypes.BALL:
    #         self.stamp()
    #     elif self.goal_position.position_type == PositionTypes.BOX:
    #         self.align()
    #     else:
    #         self.next_action()

    def handle_stamp_completion(self):
        # If stamping is successful, increment the collected balls, AND remove from the occupancy map
        self.collected_balls += 1
        self.occupancy_map.remove_ball(self.current_ball_id)


    # After depositing, reset the collected balls and update the last deposit time
    def handle_deposit_completion(self):
        self.collected_balls = 0
        self.last_deposit_time = time.time()



    # Goal setting functions
    def set_goal_to_scan_point(self):
        next_scan_point = self.scan_point_gen.get_next_scan_point()
        x, y = next_scan_point.x, next_scan_point.y
        self.goal_position.update({
            'goal': Position(x, y, 0, PositionTypes.SCAN_POINT),
            'time': time.time()
        })

    def set_goal_to_nearest_ball(self):
        ball_x, ball_y, self.current_ball_id = self.occupancy_map.get_closest_ball(self.robot_pose['x'], self.robot_pose['y'], self.robot_pose['th'])
        self.goal_position.update({
            'goal': Position(ball_x, ball_y, 0, PositionTypes.BALL),
            'time': time.time()
        })

    def set_goal_to_deposit_box(self):
        self.goal_position.update({
            'goal': Position(self.collection_box_location[0], self.collection_box_location[1], 0, PositionTypes.BOX),
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

    def is_deposit_box_reached(self):
        # insert logic to check if robot is at deposit box. For now, just return True

        return True



    def is_ball_in_front(self):
        # insert logic to check if ball is in front. For now, just return True
        # pass this from coordinator, using
        return True

# Example usage
if __name__ == '__main__':

    def consume(command_q: StatefulCommandQueue):
        if not command_q.empty():
            command = command_q.get()
            print(f"----- Consumed command: {command}")

    manager = mp.Manager()
    shared_data = {
        'running': True,
        'robot_command': StatefulCommandQueue(manager),
        'vision_command': StateWrapper(manager, VisionCommands, VisionCommands.DETECT_BALL),
    }
    detection_results_q = manager.Queue()
    robot_pose = manager.dict({'x': 0.0, 'y': 0.0, 'th': 0.0})
    goal_position = manager.dict({'goal': Position(0.0, 0.0, 0.0, PositionTypes.SCAN_POINT), 'time': time.time()})
    occupancy_map = OccupancyMap()

    decision_maker = DecisionMaker(
        robot_pose=robot_pose,
        goal_position=goal_position,
        detection_results_q=detection_results_q,
        command_queue=shared_data['robot_command'],
        occupancy_map=occupancy_map
    )

    @dataclass
    class DetectionMock:
        x: float  # +X is in FRONT
        y: float  # +Y is to the RIGHT
        angle: float  # Radians, anticlockwise is positive, angle from X axis
        total_distance: float
        confidence: float
        in_collection_zone: bool

        @property
        def coords(self):
            return self.x, self.y, self.angle

    counter = 0
    while True:
        print('-------------------')
        decision_maker.update()
        if counter % 2 == 0:
            consume(shared_data['robot_command'])
        if counter % 5 == 0:
            occupancy_map.update([DetectionMock(x=counter, y=counter, angle=0, total_distance=0, confidence=0.9, in_collection_zone=False)])
        counter += 1
        time.sleep(2)  # Add a small delay to prevent busy-waiting
