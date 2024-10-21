"""
Commannd to run code
python run_slurm_parallel.py -c 80 -p "ac_dm_6.py -v -s"
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

parser.add_argument('--json', '-j', type=str, nargs='+', help='Json Files', required=True)  # the json configuration
parser.add_argument('--list', '-l', type=bool, default = False)

args = parser.parse_args()
json_files = args.json
pythoncommands = []
for json_file in json_files:
    print(json_file)
    d = get_sorted_dict(json_file)
    experiments = get_param_iterable(d)

    pending_experiments = get_list_pending_experiments(experiments)
    if not args.list:
        print(f"Experiments : {json_file} : {len(pending_experiments)} ")
    else:
        print(f"Experiments : {json_file} : {len(pending_experiments)} :  {pending_experiments}")

