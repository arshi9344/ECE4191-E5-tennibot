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
            command_queue, # the shared command cmd_queue inside shared_data
            occupancy_map, # shared object instantiated inside ProcessCoordinator
            deposit_time_limit=8*60,
            max_capacity=4,
            collection_box_location=(COURT_XLIM, -COURT_YLIM +0.5)
    ):
        # Shared multiprocessing manager variables
        self.robot_pose = robot_pose  # Shared dict with robot's current pose {'x': x, 'y': y, 'th': th}
        self.goal_position = goal_position  # Shared dict for the goal position
        self.command_queue : StatefulCommandQueue = command_queue # Shared command cmd_queue

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
        self.scan_point_gen = ScanPointGenerator(flip_x = False, flip_y = True)
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
            {'name': RobotStates.DEPOSIT, 'on_enter': '', 'on_exit': ''},  # need to issue RobotCommands.DEPOSIT command
        ]

        transitions = [
            ##### These are the transitions that determine the robot's next action based on its internal state #####
            # all the methods under 'conditions' must return True for the transition to occur
            # all the methods under 'unless' must return False for the transition to occur

            # Transitions for driving to a location
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.IDLE, RobotStates.DRIVE_TO_SCAN_POINT, RobotStates.DEPOSIT, RobotStates.ROTATE_SCAN],
                'dest': RobotStates.DRIVE_TO_BALL,
                'conditions': ['has_known_balls'], # NOTABLY, there is NO _check_command_completion here, because the robot should immediately go back to deposit box
                'unless': ['should_return_to_deposit']
            },
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.DRIVE_TO_BALL, RobotStates.DRIVE_TO_SCAN_POINT, RobotStates.STAMP, RobotStates.ROTATE_SCAN], # i.e. NOT IDLE, DEPOSIT, ALIGN
                'dest': RobotStates.DRIVE_TO_DEPOSIT_BOX,
                'conditions': ['should_return_to_deposit']  # NOTABLY, there is NO _check_command_completion here, because the robot should immediately go back to deposit box
            },
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.IDLE, RobotStates.ROTATE_SCAN, RobotStates.DEPOSIT], # could add RobotStates.STAMP, but permitted only if another ball is within 50cm or something
                'dest': RobotStates.DRIVE_TO_SCAN_POINT,
                'conditions': ['_check_command_completion'],
                'unless': ['has_known_balls', 'should_return_to_deposit']
            },

            # Transitions for performing an action (stamp, align, deposit, rotate_scan)
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.DRIVE_TO_BALL, RobotStates.STAMP],
                'dest': RobotStates.STAMP,
                'conditions': ['_check_command_completion'],
                'unless': ['should_return_to_deposit']
            },
            {
                'trigger': 'decide_next_action',
                'source': RobotStates.DRIVE_TO_DEPOSIT_BOX,
                'dest': RobotStates.ALIGN,
                'conditions': ['should_return_to_deposit','is_deposit_box_reached', '_check_command_completion'],
            },
            {
                'trigger': 'decide_next_action',
                'source': RobotStates.ALIGN,
                'dest': RobotStates.DEPOSIT,
                'conditions': ['_check_command_completion']
            },
            {
                'trigger': 'decide_next_action',
                'source': [RobotStates.DRIVE_TO_SCAN_POINT, RobotStates.STAMP],
                'dest': RobotStates.ROTATE_SCAN,
                'conditions': ['_check_command_completion'] #
            },

            ##### manual transitions for testing #####
            # IDLE = 0  # Doing nothing
            # DRIVE_TO_BALL = 1  # implemented
            # DRIVE_TO_DEPOSIT_BOX = 2
            # DRIVE_TO_SCAN_POINT = 3  # Exploring the court.
            # ROTATE_SCAN = 4
            # STAMP = 5
            # ALIGN = 6
            # DEPOSIT = 7
            {'trigger': 'manual_idle', 'source': '*', 'dest': RobotStates.IDLE },
            {'trigger': 'manual_drive_to_ball', 'source': '*', 'dest': RobotStates.DRIVE_TO_BALL, 'conditions': ['_check_command_completion']},
            {'trigger': 'manual_drive_to_deposit_box', 'source': '*', 'dest': RobotStates.DRIVE_TO_DEPOSIT_BOX, 'conditions': ['_check_command_completion']},
            {'trigger': 'manual_drive_to_scan_point', 'source': '*', 'dest': RobotStates.DRIVE_TO_SCAN_POINT, 'conditions': ['_check_command_completion']},
            {'trigger': 'manual_rotate_scan', 'source': '*', 'dest': RobotStates.ROTATE_SCAN, 'conditions': ['_check_command_completion']},
            {'trigger': 'manual_stamp', 'source': '*', 'dest': RobotStates.STAMP, 'conditions': ['_check_command_completion']},
            {'trigger': 'manual_align', 'source': '*', 'dest': RobotStates.ALIGN, 'conditions': ['_check_command_completion']},
            {'trigger': 'manual_deposit', 'source': '*', 'dest': RobotStates.DEPOSIT, 'conditions': ['_check_command_completion']},

        ]

        # Initialize the state machine
        self.machine = Machine(model=self, states=states, transitions=transitions, initial=RobotStates.IDLE, after_state_change='_issue_command')


    def update(self, verbose=False):
        """Main update called in ProcessCoordinator control loop."""
        # print(f"Queue: {self.command_queue.show_entire_queue()}")
        if verbose: print(f'UPDATE: State before update (AKA what doing now): {self.state}')
        self._refresh_ball_goal() # this is to update the ball goal position to the nearest ball using most recent perception data

        self._check_deposit_completion() # this is to check if the robot has finished depositing balls.
        ## The reason we can't use an 'after' transition for the above is is because we need to check this every loop iteration and reset the self.last_deposit_time and self.collected_balls,
        ## otherwise the robot will never leave the deposit state.

        self._check_command_completion(verbose=False) # JUST FOR DEBUG
        self.decide_next_action() # method added by the state machine at runtime

        #TODO: Refine state completion check logic here


    def _issue_command(self):
        # issue the command to the command cmd_queue
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

        print(f"NEW STATE: {self.state}, issued command_id: {self.curr_command_id}")


    # Check if the last issued A.K.A current command is completed or still processing. This is used to determine if the next robotstate should be entered,
    # and therefore, if a new command should be issued.
    # This must return true for the state machine to transition to the next state, for all states, EXCEPT the DRIVE_TO_DEPOSIT_BOX state, as once the timer
    # has expired, the robot should immediately go back and start depositing balls.

    def _check_command_completion(self, verbose=False) -> bool:
        return_val = False
        if self.state == RobotStates.IDLE:
            # If we're in IDLE state, then we can always transition to the next state
            return_val = True

        elif self.curr_command_id is not None:
            status = self.command_queue.get_status(self.curr_command_id)
            if status == CommandStatus.DONE:
                # self.curr_command_id = None # Removed this because I don't want to change state here, seems messy
                return_val = True
            elif status == CommandStatus.FAILED:
                # self.curr_command_id = None
                return_val = True
            elif status == CommandStatus.PROCESSING:
                return_val = False
            elif status == CommandStatus.QUEUED:
                return_val = False
            if verbose: print(f"    _check_command_completion: {self.state} with {self.command_queue.get_data(self.curr_command_id)} (id: {self.curr_command_id}) is {status}")
        else:
            if verbose: print(f"_check_command_completion: No previous command")
            return_val = True  # If there is no previous command, return True, because there is nothing to wait for

        return return_val

    def _check_deposit_completion(self):
        if self.curr_command_id is not None and self.state == RobotStates.DEPOSIT and self._check_command_completion():
            # After depositing, reset the collected balls and update the last deposit time
            self.collected_balls = 0
            self.last_deposit_time = time.time()

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
        print(f"Setting goal to deposit box: {self.collection_box_location}")
        self.goal_position.update({
            'goal': Position(self.collection_box_location[0], self.collection_box_location[1], 0, PositionTypes.BOX),
            'time': time.time()
        })

    ####### State checks, truth conditions #######

    # Returns True if occupancy map is not empty
    def has_known_balls(self):
        return not self.occupancy_map.is_empty()

    # Returns True if robot should return to deposit box. Depends on time elapsed and capacity.
    def should_return_to_deposit(self, verbose=False):
        time_condition = (time.time() - self.last_deposit_time) > self.deposit_time_limit
        capacity_condition = self.collected_balls >= self.max_capacity and self.collected_balls > 0

        if time_condition or capacity_condition:
            if verbose: print(f"Returning to deposit box; time condition: {time_condition}, capacity condition: {capacity_condition}")
            return True
        return False

    def is_deposit_box_reached(self):
        # insert logic to check if robot is at deposit box. For now, just return True
        # we're also relying on the check_command_completion method to ensure the robot has finished processing the
        # previous command, so this function is just for any additional checks immediately prior to depositing. not sure if we need this.
        # TODO: Decide if we need this or not. If not, just leave the return True
        return True



    def is_ball_in_front(self):
        # insert logic to check if ball is in front. For now, just return True
        # pass this from coordinator, using
        # TODO: Add logic to see if ball is in front. DecisionMaker may need perception results? hmmm
        return False

# Example usage
if __name__ == '__main__':
    def mock_robot(command_q: StatefulCommandQueue, last_got_command_id):
        # Mark previous command as Done
        if last_got_command_id is not None:
            cmd_name = command_q.get_data(last_got_command_id)
            command_q.mark_done(last_got_command_id)
            print(f"----- Mock robot DONE {cmd_name}, id: {last_got_command_id}")
            return None

        # Get the current command
        elif not command_q.empty():
            command, cmd_id = command_q.get()
            print(f"### Mock robot GOT {command}, id: {cmd_id}")
            return cmd_id


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
        command_queue=shared_data['robot_command'],
        occupancy_map=occupancy_map,
        deposit_time_limit=15
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
    last_got_cmd_id = None
    while True:
        print('-------------------')
        decision_maker.update()
        print(f"OCC. MAP: {occupancy_map}")
        if counter % 2 == 0:
            last_got_cmd_id = mock_robot(shared_data['robot_command'], last_got_cmd_id)
        if counter % 5 == 0 and counter > 0:
            print(f"### ADDED BALL")
            occupancy_map.update([DetectionMock(x=counter/5, y=counter/5, angle=0, total_distance=0, confidence=0.9, in_collection_zone=False)])
        counter += 1
        time.sleep(1)  # Add a small delay to prevent busy-waiting
