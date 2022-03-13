from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import numpy as np
import re
import os
import pdb
import matplotlib.pyplot as plt
#import argparse

# finish this off later.

#parser = argparse.ArgumentParser()
#parser.parse_args()

FIGSIZE=(9,5)

regex = re.compile('^events.out.tfevents*')

hidden_dims = [8, 16, 32, 64, 100, 128]#, 256]#, #200, 256]#, 512]
styles = ["reformer", "original", "lsh"]

SCALARS_OF_INTEREST = ["train/MEMORY", "train/TIME"]
nice_name = ["memory per batch (GB)", "time per batch (seconds)"]


def get_log_file_path(dims, style):
    log_dir = "save/train/memorytest-{}-{}-01".format(dims, style)
    possible_log_dirs = glob.glob("save/train/memorytest-{}-{}-[0-9]*".format(dims,style)) 
    max_num = 0
    for possible_log_dir in possible_log_dirs:
        if int(possible_log_dir.split("-")[-1]) > max_num:
            log_dir = possible_log_dir
    for _, _, files in os.walk(log_dir):
        for f in files:
            if regex.match(f):
                return os.path.join(log_dir, f)
    return None

for style in styles:
    scalars = {scalar: list() for scalar in SCALARS_OF_INTEREST}
    for dims in hidden_dims:
        print("{}, {}".format(style, dims))
        try:
            log_file_path = get_log_file_path(dims, style)
            event_acc = EventAccumulator(log_file_path)
            event_acc.Reload() # necessary to load in the dataa
            
            for SCALAR in SCALARS_OF_INTEREST:
                event = event_acc.Scalars(SCALAR)[0] # just use the first one
                scalar = event.value
                scalars[SCALAR].append(scalar)
        except:
            print('exception for {}, {}'.format(style, dims))
    print(scalars)
    for i,scalar in enumerate(scalars):
        plt.figure(num=i, figsize=FIGSIZE)
        plt.plot(hidden_dims[:len(scalars[scalar])], np.array(scalars[scalar]), label=style)
for i,scalar in enumerate(SCALARS_OF_INTEREST):
    plt.figure(num=i, figsize=FIGSIZE) 
    plt.legend()
    plt.ylabel(nice_name[i], fontsize=16)
    plt.xlabel('hidden dimensions', fontsize=16)
    plt.savefig('{}.png'.format(scalar.split("/")[-1]))
