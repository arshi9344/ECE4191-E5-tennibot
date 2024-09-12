
#%%
import multiprocessing as mp
import time
import os
import psutil


def cpu_bound_task(process_name):
    # Get the current process ID
    pid = os.getpid()

    # Get the CPU core this process is running on
    process = psutil.Process(pid)

    print(f"{process_name} (PID: {pid}) started")

    # Perform a CPU-bound task
    start_time = time.time()
    while time.time() - start_time < 10:  # Run for 10 seconds
        _ = [i ** 2 for i in range(10000)]

        # Check CPU usage periodically
        if time.time() - start_time > 1:  # After 1 second of processing
            cpu_num = process.cpu_num()
            print(f"{process_name} (PID: {pid}) running on CPU core: {cpu_num}")
            break