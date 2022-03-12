from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

# finish this off later.

parser = argparse.ArgumentParser()
parser.parse_args()




log_file_path = ""
SCALARS_OF_INTEREST = ["train/MEMORY", "train/TIME"]
event_acc = EventAccumulator(log_file_path)

for SCALAR in SCALARS_OF_INTEREST:
    """
    >>> event_acc.Scalars('train/MEMORY')
[ScalarEvent(wall_time=1647074191.7343554, step=16, value=932629504.0), ScalarEvent(wall_time=1647074197.6242537, step=32, value=1865259008.0), ScalarEvent(wall_time=1647074204.2224517, step=48, value=2797888512.0), ScalarEvent(wall_time=1647074209.802795, step=50, value=3327046656.0)]
>>> event_acc.Scalars('train/MEMORY')[0].value
    """
    for event in event_acc.Scalars(SCALAR):
       print(event.value)
       print(event.step)

