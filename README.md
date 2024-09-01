# ECE4191, Group E05 - Autonomous Tennis Ball Collector Robot

This repo is organised as follows:

- `robot_core/` and its subfolders contain the main codebase for the robot. This includes all our classes.
- `notebooks/` contains all our Jupyter notebooks for testing, development, and runtime.
  - `notebooks/run_robot/` contains notebooks that we use to run the robot
  - `notebooks/scratchspace/` contains notebooks for testing, development, and debugging
  - `notebooks/simulation/` contains notebooks for testing our codebase with a simulated robot.

## Software Overview
- Each of our classes are designed to be entirely self-contained, and can be run independently of the rest of the codebase.
- The exception to this is Orchestrator, in `robot_core/orchestration/orchestrator.py`, which provides high level logic and pipes data from one class to another. 
- Use the simulation notebooks in `notebooks/simulation/` to test the codebase without the robot.