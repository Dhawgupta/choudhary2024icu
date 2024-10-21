"""
Commannd to run code
python run_slurm_parallel.py -c 80 -p "ac_dm_6.py -v -s"
This will run the python file on everyt json file
"""
import os
import sys
import argparse

sys.path.append(os.getcwd())
import time
import numpy as np
from src.utils.run_utils import get_list_pending_experiments
from src.utils.json_handling import get_sorted_dict, get_param_iterable

parser = argparse.ArgumentParser()
parser.add_argument('--pythonfile', '-p', type=str)
parser.add_argument('--json', '-j', type=str, nargs='+', help='Json Files', required=True)  # the json configuration
parser.add_argument('--cpus', '-c', type=int, default=-1,
                    help="The number of CPU's to be used for parallel, otherwise uses all")


args = parser.parse_args()
json_files = args.json
pythoncommands = []
num_commands = len(json_files)
for json_file in json_files:

    # make the command with the json file
    com = f'python {args.pythonfile} {json_file}\n'
    pythoncommands.append(com)

nodes = 1  # we only need 1 node , the current node
command_nodes = [[] for i in range(nodes)]
# give each node command
for i, c in enumerate(pythoncommands):
    command_nodes[i % nodes].append(c)

foldername = './temp/parallel_scripts/'
if not os.path.exists(foldername):
    os.makedirs(foldername, exist_ok=True)
filename = [f'./temp/parallel_scripts/node_{i}_{str(np.random.randint(0, 100000))}.txt' for i in range(nodes)]

# write commands in files
for i, f in enumerate(filename):
    fil = open(f, 'w')
    fil.writelines(command_nodes[i])
    fil.close()

parallel_commands = []
for f in filename:
    if args.cpus < 0:
        command = 'parallel --verbose -P -0 :::: {}'.format(f)
    else:
        command = 'parallel --verbose -P {} :::: {}'.format(args.cpus, f)
    parallel_commands.append(command)

cwd = os.getcwd()

bash_file = f"#!/bin/sh\n" \
            f"cd {cwd}\n" \
            f"export PYTHONPATH={cwd}:$PYTHONPATH\n" \
            f"{parallel_commands[0]}"

print(bash_file)
os.system(f'{bash_file}')