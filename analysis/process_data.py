'''
This file will take in json files and process the data across different runs to store the summary
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import numpy as np

from src.utils.json_handling import get_sorted_dict , get_param_iterable_runs
from src.utils.formatting import create_file_name, create_folder
from analysis.utils import load_different_runs_control, pkl_saver, pkl_loader

# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/process_data.py <list of json files")
    exit()

json_files = sys.argv[1:] # all the json files

# convert all json files to dict
json_handles = [get_sorted_dict(j) for j in json_files]

def process_runs(runs):
    # get mean and std
    mean = np.mean(runs, axis = 0)
    stderr = np.std(runs , axis = 0) / np.sqrt(runs.shape[0])
    return mean , stderr


#
def process_runs_episodes(returns, discounted_returns, episodes):
    # get the smallest number of episode
    min_episodes = np.min(episodes[:,-1])
    num_runs, num_steps = returns.shape
    # print(num_runs, num_steps)
    episode_returns = np.zeros((num_runs, min_episodes))
    episode_discounted_returns = np.zeros((num_runs, min_episodes))
    for i in range(num_runs):
        for j in range(min_episodes):
            episode_returns[i,j] = returns[i, episodes[i,:] == j+1].mean()
            episode_discounted_returns[i,j] = discounted_returns[i, episodes[i,:] == j+1].mean()
    mean_returns, stderr_returns = process_runs(episode_returns)
    mean_discounted_returns, stderr_discounted_returns = process_runs(episode_discounted_returns)
    return mean_returns, stderr_returns, mean_discounted_returns, stderr_discounted_returns



# currentl doesnt not handle frames
def process_data_interface(json_handles):
    for js in json_handles:
        runs = []
        iterables = get_param_iterable_runs(js)
        for i in iterables:
            folder, file = create_file_name(i, 'processed')
            create_folder(folder) # make the folder before saving the file
            filename = folder + file + '.pcsd'
            # check if file exists
            print(filename)
            if os.path.exists(filename):
                print("Processed")

            else:
                # shape (num_seeds, num_steps) e.g., (30, 100K)
                returns, discounted_returns, num_steps = load_different_runs_control(i)
                

                # train, test, validation, loss = load_different_runs(i)
                mean_returns, stderr_returns = process_runs(returns)
                mean_discounted_returns, stderr_discounted_returns = process_runs(discounted_returns)
                mean_num_steps, stderr_num_steps = process_runs(num_steps)
                # mean_msve, stderr_msve = process_runs(msve)
                # mean_mstde, stderr_mstde = process_runs(mstde)

                # returns
                return_data = {
                    'mean' : mean_returns,
                    'stderr' : stderr_returns
                }
                # discounted returns
                discounted_return_data = {
                    'mean' : mean_discounted_returns,
                    'stderr' : stderr_discounted_returns
                }

                num_steps_data = {
                    'mean' : mean_num_steps, 
                    'stderr' : stderr_num_steps
                }


                pkl_saver({

                    'returns' : return_data,
                    'discounted_returns' : discounted_return_data,
                    'num_steps': num_steps_data
                    # 'mstde' : mstde_data
                }, filename)
                print("Saved")

    # print(iterables)
            

# process data based on episodes rather than steps
def process_data_interface_episodes_from_steps(json_handles):
    for js in json_handles:
        runs = []
        iterables = get_param_iterable_runs(js)
        for i in iterables:
            folder, file = create_file_name(i, 'processed')
            create_folder(folder) # make the folder before saving the file
            filename = folder + file + '_episodes' + '.pcsd'
            # check if file exists
            print(filename)
            if os.path.exists(filename):
                print("Processed")

            else:
                returns, discounted_returns, episodes = load_different_runs_control(i)

                # train, test, validation, loss = load_different_runs(i)
                mean_returns, stderr_returns, mean_discounted_returns, stderr_discounted_returns =  process_runs_episodes(returns, discounted_returns, episodes)
                # mean_returns, stderr_returns = process_runs(returns)
                # mean_discounted_returns, stderr_discounted_returns = process_runs(discounted_returns)
                # mean_msve, stderr_msve = process_runs(msve)
                # mean_mstde, stderr_mstde = process_runs(mstde)

                # returns
                return_data = {
                    'mean' : mean_returns,
                    'stderr' : stderr_returns
                }
                # discounted returns
                discounted_return_data = {
                    'mean' : mean_discounted_returns,
                    'stderr' : stderr_discounted_returns
                }


                pkl_saver({

                    'returns' : return_data,
                    'discounted_returns' : discounted_return_data
                    # 'mstde' : mstde_data
                }, filename)
                print("Saved")



if __name__ == '__main__':
    process_data_interface(json_handles)