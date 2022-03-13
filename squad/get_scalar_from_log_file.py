from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re
import os
import pdb
#import argparse

# finish this off later.

#parser = argparse.ArgumentParser()
#parser.parse_args()

regex = re.compile('^events.out.tfevents*')

hidden_dims = [32, 64, 128, 256, 512]
styles = ["reformer", "original", "lsh"]

SCALARS_OF_INTEREST = ["train/MEMORY", "train/TIME"]

def get_log_file_path(dims, style):
    log_dir = "save/train/memorytest-{}-{}-01".format(dims, style)
    for _, _, files in os.walk(log_dir):
        for f in files:
            if regex.match(f):
                return os.path.join(log_dir, f)
    return None

for style in styles:
    scalars = {scalar: list() for scalar in SCALARS_OF_INTEREST}
    for dims in hidden_dims:
        log_file_path = get_log_file_path(dims, style)
        event_acc = EventAccumulator(log_file_path)
        pdb.set_trace()
        
        for SCALAR in SCALARS_OF_INTEREST:
            event = event_acc.Scalars(SCALAR)[0] # just use the first one
            scalar = event.value
            scalars[SCALAR].append(scalar)
            
    print(scalars)
