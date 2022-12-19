#!/usr/bin/env python
from __future__ import print_function
import subprocess
import random

def get_unused_gpus():
    p = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = out.decode()

    lines = out.split('\n')
    for i in range(len(lines)):
        if 'Processes' in lines[i]:
            break

    ngpus = 0
    for line in lines[:i]:
        tokens = line.split()
        if len(tokens) > 1:
            gpu = tokens[1]
            try:
                gpu = int(gpu)
                if gpu+1 > ngpus:
                    ngpus = gpu+1
            except:
                pass

    lines = lines[i+1:]
    gpus_used = []
    for line in lines:
        tokens = line.split()
        if len(tokens) > 1:
            gpu = tokens[1]
            try:
                gpu = int(gpu)
                gpus_used.append(gpu)
            except ValueError:
                pass

    unused_gpus = []
    for i in range(ngpus):
        if i not in gpus_used:
            unused_gpus.append(i)
    return unused_gpus

def random_free_gpu():
    gpus = get_unused_gpus()
    if len(gpus) == 0:
        return None
    return random.choice(gpus)

if __name__ == '__main__':
    # ",".join(get_unused_gpus())
    print(random_free_gpu())
