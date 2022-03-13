from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import numpy as np
import re
import os
import pdb
import matplotlib.pyplot as plt
import wandb
#import argparse

# finish this off later.

#parser = argparse.ArgumentParser()
#parser.parse_args()

FIGSIZE=(11,5)

hidden_dims = [8, 16, 32, 64, 128, 256]#, 256]#, #200, 256]#, 512]
styles = ["reformer", "original", "lsh"]


api = wandb.Api()
ENTITY = "andrewgaut"
PROJECT = "inquisition-squad"
VALUES = ["system.gpu.0.memoryAllocated"]
NAMES = ["GPU Memory Allocation (%)"]

stored_values = [dict() for _ in range(len(VALUES))] 

runs = api.runs(ENTITY+ "/" + PROJECT)
for run in runs:
    split = run.name.split("-")
    style = split[1]
    dims = split[2]

    history = run.history(stream="system")

    for i, stored_value in enumerate(stored_values):
        if style not in stored_value:
            stored_value[style] = dict()
        stored_value[style][dims] = history[VALUES[i]][0]






for style in styles:
    for i,value in enumerate(VALUES):
        sorted_keys = list(stored_values[i][style].keys())
        sorted_keys.sort(key=lambda x: int(x))
        plot_vals = [stored_values[i][style][key] for key in sorted_keys] 

        plt.figure(num=i, figsize=FIGSIZE)
        plt.plot(sorted_keys, np.array(plot_vals), label=style)

for i,value in enumerate(VALUES):
    plt.figure(num=i, figsize=FIGSIZE) 
    plt.legend()
    plt.ylabel(NAMES[i], fontsize=16)
    plt.xlabel('hidden dimensions', fontsize=16)
    plt.savefig('{}.png'.format(value.split(".")[-1]))
