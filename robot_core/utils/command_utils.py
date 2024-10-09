import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from robot_core.coordinator.commands import RobotCommands
import time

class CommandStatus(Enum):
    QUEUED = 0
    PROCESSING = 1 # not using this for now
    DONE = 2
    FAILED = 3 # not using this for now

@dataclass
class Command:
    id: int
    data: RobotCommands
    status: CommandStatus
    issue_time: float

class StatefulCommandQueue:
    def __init__(self, manager: mp.Manager):
        self.MAX_QUEUE_SIZE = 2
        self.queue = manager.list()
        self.command_map = manager.dict()
        self.command_id_counter = manager.Value('i', 0)
        self.most_recent_issued_command = None

    # Main method for adding commands to the queue. Use put when issuing commands to robot.
    def put(self, command_data: RobotCommands):
        if len(self.queue) >= self.MAX_QUEUE_SIZE:
            raise ValueError(f"Command queue full, MAX_QUEUE_SIZE reached ({self.MAX_QUEUE_SIZE})")

        self.command_id_counter.value += 1
        command_id = self.command_id_counter.value
        command = Command(
            id=command_id,
            data=command_data,
            status=CommandStatus.QUEUED,
            issue_time=time.time()
        )
        self.queue.append(command)
        self.command_map[command_id] = command
        self.most_recent_issued_command = command
        return command_id

    # Main method for getting commands to the queue. Use get when getting commands.
    def get(self) -> Optional[RobotCommands]:
        if self.queue:
            command = self.queue.pop(0)
            command.status = CommandStatus.PROCESSING
            self.command_map[command.id] = command
            return command.data
        return None

    # Mark a command as done, using a command id.
    def mark_done(self, command_id: int):
        if command_id in self.command_map:
            self.command_map[command_id].status = CommandStatus.DONE

    # Mark a command as failed, using a command id.
    def mark_failed(self, command_id: int):
        if command_id in self.command_map:
            self.command_map[command_id].status = CommandStatus.FAILED

    # Get the status of a command using the id.
    def get_status(self, command_id: int) -> Optional[CommandStatus]:
        if command_id in self.command_map:
            return CommandStatus(self.command_map[command_id].status)
        return None

    # Get the data of a previously issued command.
    def get_data(self, command_id: int) -> Optional[RobotCommands]:
        if command_id in self.command_map:
            return self.command_map[command_id].data
        return None

    # Get all commands that are processing
    def get_processing_commands(self):
        return [cmd for cmd in self.command_map.values() if cmd.status == CommandStatus.PROCESSING]

    # Get all commands that are completed
    def remove_completed(self):
        self.command_map = {cmd_id: cmd for cmd_id, cmd in self.command_map.items()
                            if CommandStatus(cmd.status) != CommandStatus.DONE}

    # Remove all commands in queue. Shouldn't really neeed to use this, only here in case.
    def remove_all(self):
        del self.queue[:]
        self.command_map.update({})

    # Get the last issued command. Shouldn't really need this.
    def last_issued(self) -> int:
        if self.most_recent_issued_command is None:
            return -1
        return self.most_recent_issued_command

    def __len__(self):
        return len(self.queue)

    def empty(self):
        return len(self.queue) == 0




if __name__ == "__main__":
    # Example usage remains the same as in the previous version
    import random
    import time


    def producer(queue):
        for _ in range(10):
            new_state = random.choice(list(RobotCommands))
            cmd_id = queue.put(new_state)
            print(f"Producer: Added command with id {cmd_id}, state: {new_state.name}")
            time.sleep(0.5)


    def consumer(queue):
        while True:
            command_data = queue.get()
            if command_data is None:
                time.sleep(0.1)
                continue

            print(f"Consumer: Processing command: {command_data.name}")
            time.sleep(random.uniform(1, 3))  # Simulate processing time

            # Since we're not using PROCESSING or FAILED states, we'll always mark as DONE
            queue.mark_done(queue.command_id_counter.value)
            print(f"Consumer: Completed command {queue.command_id_counter.value}")


    def monitor(queue):
        while True:
            processing = queue.get_processing_commands()
            print(f"Monitor: Currently processing commands: {[cmd.id for cmd in processing]}")
            queue.remove_completed()
            time.sleep(1)



    manager = mp.Manager()
    queue = StatefulCommandQueue(manager)

    producer_process = mp.Process(target=producer, args=(queue,))
    consumer_process = mp.Process(target=consumer, args=(queue,))
    monitor_process = mp.Process(target=monitor, args=(queue,))

    producer_process.start()
    consumer_process.start()
    monitor_process.start()

    producer_process.join()
    time.sleep(5)  # Allow some time for remaining commands to be processed

    # Demonstrate remove_all functionality
    print("Removing all commands from the queue...")
    queue.remove_all()
    print(f"Commands remaining in queue: {len(queue)}")

    consumer_process.terminate()
    monitor_process.terminate()