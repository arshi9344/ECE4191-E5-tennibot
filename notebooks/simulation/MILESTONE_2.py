import os, sys
import matplotlib.pyplot as plt
try: # Import the GPIO library. If it fails, we assume we are running on a non-Raspberry Pi system.
    import RPi.GPIO as GPIO
    import gpiozero
except ImportError:
    print("Not running on Raspberry Pi. GPIO library not imported.")
    pass
# %matplotlib inline
plt.ion()  # Turn on interactive mode
current_dir = os.getcwd()
repo_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
# print(repo_dir)
sys.path.append(repo_dir)


from robot_core.coordinator.ProcessCoordinator import Coordinator

###############################################################################
###############################################################################
# In robot coordinate space
# +X IN FRONT OF THE ROBOT 
# +Y TO LEFT OF ROBOT


###############################################################################
###############################################################################



if __name__ == "__main__":
    coordinator = Coordinator(
        simulate=True, 
        live_graphs=True, 
        graph_interval=4, 
        log=False, 
        clear_output=False, 
        plot_time_window=5,
        efficient_plotting=True,
        save_figs=False,
        deposit_time_limit=8*60, # seconds
        max_ball_capacity=5
    )
    
    coordinator.run()
