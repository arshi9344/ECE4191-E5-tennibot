import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple
import random
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

    # Main method for adding commands to the cmd_queue. Use put when issuing commands to robot.
    # Not using put_with_replace for now
    def put(self, command_data: RobotCommands, put_with_replace=False) -> int:
        if len(self.queue) >= self.MAX_QUEUE_SIZE:
            raise ValueError(f"Command cmd_queue full, MAX_QUEUE_SIZE reached ({self.MAX_QUEUE_SIZE})")

        self.command_id_counter.value += 1
        command_id = self.command_id_counter.value
        command = Command(
            id=command_id,
            data=command_data,
            status=CommandStatus.QUEUED,
            issue_time=time.time()
        )
        if put_with_replace: # if True, immediately replace the command in the queue
            self.remove_all()
        self.queue.append(command)
        self.command_map[command_id] = command
        self.most_recent_issued_command = command
        return command_id

    # Main method for getting commands to the cmd_queue. Use get when getting commands.
    def get(self) -> Optional[Tuple[RobotCommands, int]]:
        if self.queue:
            command = self.queue.pop(0)
            command.status = CommandStatus.PROCESSING
            self.command_map[command.id] = command
            return command.data, command.id
        return None

    # Mark a command as done, using a command id.
    def mark_done(self, command_id: int):
        if command_id in self.command_map:
            old_command = self.command_map[command_id]

            # Create a new Command object with the updated status
            updated_command = Command(
                id=old_command.id,
                data=old_command.data,
                status=CommandStatus.DONE,
                issue_time=old_command.issue_time
            )
            # Replace the old command in the command_map with the updated one

            self.command_map[command_id] = updated_command
            # print(f"Command {command_id} MARKED DONE: {updated_command.status}")

    # Mark a command as failed, using a command id.
    def mark_failed(self, command_id: int):
        if command_id in self.command_map:
            old_command = self.command_map[command_id]

            # Create a new Command object with the updated status
            updated_command = Command(
                id=old_command.id,
                data=old_command.data,
                status=CommandStatus.FAILED,
                issue_time=old_command.issue_time
            )

            # Replace the old command in the command_map with the updated one
            self.command_map[command_id] = updated_command
            # print(f"Command {command_id} MARKED FAILED: {updated_command.status}")

    # Get the status of a command using the id.
    def get_status(self, command_id: int) -> Optional[CommandStatus]:
        if command_id in self.command_map:
            return CommandStatus(self.command_map[command_id].status)
        return None

    # Get the data of a previously issued command. This is different from .get(), as it doesn't remove the command.
    def get_data(self, command_id: int) -> Optional[RobotCommands]:
        if command_id in self.command_map:
            return self.command_map[command_id].data
        return None

    # Get all commands that are processing
    def get_processing_commands(self):
        return [cmd for cmd in self.command_map.values() if cmd.status == CommandStatus.PROCESSING]

    def show_entire_queue(self):
        return self.command_map

    # Get all commands that are completed
    def remove_completed(self):
        self.command_map = {cmd_id: cmd for cmd_id, cmd in self.command_map.items()
                            if CommandStatus(cmd.status) != CommandStatus.DONE}

    # Remove all commands in cmd_queue. Shouldn't really neeed to use this, only here in case.
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



def mock_producer(cmd_queue):
    for _ in range(5):  # Reduced to 5 commands to stay within MAX_QUEUE_SIZE
        new_command = random.choice(list(RobotCommands))
        try:
            cmd_id = cmd_queue.put(new_command)
            print(f"Producer: Added command with id {cmd_id}, command: {new_command.name}")
        except ValueError as e:
            print(f"Producer: Failed to add command - {str(e)}")
        time.sleep(0.5)


def mock_consumer(cmd_queue):
    while True:
        result = cmd_queue.get()
        if result is None:
            time.sleep(0.1)
            continue

        command_data, cmd_id = result
        print(f"Consumer: Processing command id {cmd_id}: {command_data.name}")
        time.sleep(random.uniform(1, 2))  # Simulate processing time

        if random.random() < 0.8:  # 80% chance of success
            cmd_queue.mark_done(cmd_id)
            print(f"Consumer: Completed command {cmd_id}")
        else:
            cmd_queue.mark_failed(cmd_id)
            print(f"Consumer: Failed to complete command {cmd_id}")


def mock_monitor(cmd_queue):
    while True:
        processing = cmd_queue.get_processing_commands()
        print(f"Monitor: Currently processing commands: {[cmd.id for cmd in processing]}")

        for cmd_id in cmd_queue.command_map.keys():
            status = cmd_queue.get_status(cmd_id)
            print(f"Monitor: Command {cmd_id} status: {status.name if status else 'Unknown'}")

        cmd_queue.remove_completed()
        print(f"Monitor: Queue size after removing completed: {len(cmd_queue)}")
        time.sleep(1)


if __name__ == "__main__":


    manager = mp.Manager()
    queue = StatefulCommandQueue(manager)

    producer_process = mp.Process(target=mock_producer, args=(queue,))
    consumer_process = mp.Process(target=mock_consumer, args=(queue,))
    monitor_process = mp.Process(target=mock_monitor, args=(queue,))

    producer_process.start()
    consumer_process.start()
    monitor_process.start()

    producer_process.join()
    time.sleep(5)  # Allow some time for remaining commands to be processed

    # Demonstrate additional functionality
    print("\nTesting additional functionality:")
    print(f"Last issued command: {queue.last_issued().id if queue.last_issued() != -1 else 'None'}")

    if not queue.empty():
        last_cmd_id = queue.last_issued().id
        print(f"Status of last command: {queue.get_status(last_cmd_id).name}")
        print(f"Data of last command: {queue.get_data(last_cmd_id).name}")

    print("\nRemoving all commands from the cmd_queue...")
    queue.remove_all()
    print(f"Commands remaining in cmd_queue: {len(queue)}")

    consumer_process.terminate()
    monitor_process.terminate()