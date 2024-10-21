"""
Deletes all experiments corresponding to the experiment files
Usage:
python run/delete_experiment.py -j <json_files.json>
"""

import os
import sys
import argparse
sys.path.append(os.getcwd())
import time
import numpy as np
from src.utils.run_utils import get_list_pending_experiments
from src.utils.json_handling import get_sorted_dict, get_param_iterable, get_param_iterable_runs
from src.utils.formatting import create_file_name

parser = argparse.ArgumentParser()

parser.add_argument('--json', '-j', type = str ,nargs='+', help='Json Files', required=True) # the json configuration
parser.add_argument('--processed', '-p',  action='store_true',default=False, help='Delete files from processes folder')
parser.add_argument('--result', '-r',  action='store_true',default=False, help='Delete files from results folder')

args = parser.parse_args()
json_files = args.json
delete_pro = args.processed
delete_result = args.result

json_handles = [get_sorted_dict(j) for j in json_files]

# if we have to delete processesd
if delete_pro:
    for js in json_handles:
        # print(js)
        iterables = get_param_iterable_runs(js)
        for i in iterables:
            folder, file = create_file_name(i, 'processed')
            filename = folder + file + '.pcsd'
            if os.path.exists(filename):
                os.remove(filename)
                print(f'{filename} Removed')





# if we have to delete result
if delete_result:
    for json_file in json_files:
        d = get_sorted_dict(json_file)
        experiments = get_param_iterable(d)
        for exp in experiments:
            folder, filename = create_file_name(exp)
            # create experiment
            output_filename = folder + filename + '.dw'
            if os.path.exists(output_filename):
                os.remove(output_filename)
                print(f'{output_filename} Removed')

