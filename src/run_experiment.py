# make this into a file which uses argparse to run experiments
import os, sys, time
sys.path.append(os.getcwd())

import torch
import numpy as np
import matplotlib.pyplot as plt
from analysis.utils import pkl_saver
from src.algos.registry import get_algo

from src.utils.formatting import create_file_name
import types


def get_output_filename(ex):
    folder, filename = create_file_name(ex)
    if not os.path.exists(folder):
        time.sleep(2)
        try:
            os.makedirs(folder)
        except:
            pass
    output_file_name = folder + filename
    return output_file_name
    

def run_experiment(params: dict = None, force_run: bool = False):
    use_tensorboard = params.get('use_tensorboard', False)
    use_wandb = params.get('track', False)
    params.pop('use_tensorboard', None)
    params.pop('track', None)
    print("Using Tensorboard: ", use_tensorboard)
    print("Using Wandb: ", use_wandb)
    
    output_file_name = get_output_filename(params)
    args = types.SimpleNamespace(**params)

    if not force_run:
        if os.path.exists(output_file_name + '.dw'):
            print("Already Done")
            exit()

    algo = args.algo
    algo_func = get_algo(algo)

    # separate out params for storing tensorboard and wandb
    start_time = time.time()
    returns_, discounted_returns_ , num_steps = algo_func(args, use_tensorboard, use_wandb)
    finish_time = time.time()
    
   
    # import matplotlib.pyplot as plt
    # plt.plot(returns_, label='Returns')
    # plt.plot(discounted_returns_, label='Discounted Returns')
    # plt.legend()
    # plt.savefig('fig.png')
    save_stats = {
        'returns': returns_,
        'discounted_returns': discounted_returns_,
        'num_steps': num_steps,
    }
    
    
    pkl_saver(save_stats, output_file_name + '.dw')

    print(f"Finished {output_file_name} in {finish_time - start_time} seconds")



    

    


