'''
Plotting Utilities for Python
'''
import numpy as np
import torch
import os, sys
import pickle as pkl

sys.path.append(os.getcwd())

from src.utils.formatting import create_file_name
from src.utils.json_handling import get_param_iterable_runs, get_param_iterable
import itertools


def pkl_loader(filename):
    with open(filename, 'rb') as fil:
        data = pkl.load(fil)
    return data


def pkl_saver(obj, filename):
    with open(filename, 'wb') as fil:
        pkl.dump(obj, fil)


def smoothen_runs(data, factor=0.9):
    datatemp = data.reshape(-1)
    smooth_data = np.zeros_like(datatemp)
    smooth_data[0] = datatemp[0]
    for i in range(1, len(datatemp)):
        smooth_data[i] = smooth_data[i - 1] * factor + (1 - factor) * datatemp[i]
    return smooth_data


def fill_runs(exp_config, run):
    while len(run) != exp_config['epochs'] + 1:
        run.append(run[-1])
    return run


    
def load_different_runs_control(json_handle):
    '''
    Format for json handle : Should not have any lists except for seed list to load different runs
    '''
    discounted_returns = []
    returns = []
    num_steps = []

    # get the list of params
    iterable = get_param_iterable(json_handle)
    for i in iterable:
        # print(i)
        # print(i)
        # remove the tenoarbdo noard sutff
        i.pop('use_tensorboard', None)
        i.pop('track', None)
        folder, file = create_file_name(i)
        filename = folder + file + '.dw'
        # if not os.path.exists(filename):
        #     print('run experiment')
        # arr = pkl_loader(filename)
        # # losses.append(arr['losses'])
        # returns.append(arr['returns'])
        # discounted_returns.append(arr['discounted_returns'])
        # load the file
        try:
            if not os.path.exists(filename):
                print('run experiment')
            arr = pkl_loader(filename)
            # losses.append(arr['losses'])
            returns.append(arr['returns'])
            discounted_returns.append(arr['discounted_returns'])
            num_steps.append(arr['num_steps'])
        # msve.append(arr['msve'])
        # mstde.append(arr['mstde'])

            # returns.append(fill_runs(i, arr['returns']))

        except:
            print('Run not valid')
        #     pass
    returns = np.array(returns)
    discounted_returns = np.array(discounted_returns)
    num_steps = np.array(num_steps)
    # msve = np.array(msve)
    # mstde = np.array(mstde)
    return returns, discounted_returns, num_steps

def find_best_episode(json_handle, data='mstde', key=None, metric='auc', minmax='min'):
    '''
    Format of json handle : Should have list for all the parameters we want to search across
    key : If Key is none, it will find the best across all parameters else it will find the best across the key poarameter
    metric : auc or end (end :means the last 10 %  performance)
    data : Find best wrt to the following parameter
    minmax : min or max (i.e. is min better or max better)
    '''
    # swtich to positive for loss
    if minmax == 'min':
        best_auc = +np.inf
    elif minmax == 'max':
        best_auc = -np.inf
    else:
        raise Exception(f"minmax should be min or max, not {minmax}")
    best_params = None
    best_run = {}
    best_data = {}
    iterable = get_param_iterable_runs(json_handle)
    # iterable of all params
    # print(iterable)
    for i in iterable:
        folder, file = create_file_name(i, 'processed')
        filename = folder + file + '_episodes' +  '.pcsd'

        if not os.path.exists(filename):
            print(i)
            raise Exception(f"Processed Data  File Not Found, please process data first for : {filename}")
        # load the data
        data_obj = pkl_loader(filename)

        # data = torch.load(filename)
        mean = data_obj[data]['mean']
        stderr = data_obj[data]['stderr']
        # auc = np.mean(mean[-200:])
        if metric == 'auc':
            auc = np.mean(mean)
        else:
            # take mean of the last 10 % of the data
            try:
                auc = np.mean(mean[-( mean.shape[0] // 10):])
            except: # consider the cases where the run was not successfull
                auc = -np.inf            
            # auc = np.mean(mean[-200:])
        if minmax == 'min':
            if auc < best_auc:
                best_auc = auc
                best_run['mean'] = mean
                best_run['stderr'] = stderr
                best_params = i
                best_data = data_obj
        elif minmax == 'max':
            if auc > best_auc:
                best_auc = auc
                best_run['mean'] = mean
                best_run['stderr'] = stderr
                best_params = i
                best_data = data_obj

    return best_run, best_params, best_data




def find_best(json_handle, data='mstde', key=None, metric='auc', minmax='min'):
    '''
    Format of json handle : Should have list for all the parameters we want to search across
    key : If Key is none, it will find the best across all parameters else it will find the best across the key poarameter
    metric : auc or end (end :means the last 10 %  performance)
    data : Find best wrt to the following parameter
    minmax : min or max (i.e. is min better or max better)
    '''
    # swtich to positive for loss
    if minmax == 'min':
        best_auc = +np.inf
    elif minmax == 'max':
        best_auc = -np.inf
    else:
        raise Exception(f"minmax should be min or max, not {minmax}")
    best_params = None
    best_run = {}
    best_data = {}
    iterable = get_param_iterable_runs(json_handle)
    # iterable of all params
    # print(iterable)
    for i in iterable:
        folder, file = create_file_name(i, 'processed')
        filename = folder + file + '.pcsd'

        if not os.path.exists(filename):
            print(i)
            raise Exception(f"Processed Data  File Not Found, please process data first for : {filename}")
        # load the data
        data_obj = pkl_loader(filename)

        # data = torch.load(filename)
        mean = data_obj[data]['mean']
        stderr = data_obj[data]['stderr']
        # auc = np.mean(mean[-200:])
        if metric == 'auc':
            auc = np.mean(mean)
        else:
            try:
                auc = np.mean(mean[-( mean.shape[0] // 10):])
            except: # consider the cases where the run was not successfull
                auc = -np.inf   
        if minmax == 'min':
            if auc < best_auc:
                best_auc = auc
                best_run['mean'] = mean
                best_run['stderr'] = stderr
                best_params = i
                best_data = data_obj
        elif minmax == 'max':
            if auc > best_auc:
                best_auc = auc
                best_run['mean'] = mean
                best_run['stderr'] = stderr
                best_params = i
                best_data = data_obj

    return best_run, best_params, best_data

# FIXME fix this script for validatoin data
def find_best_key(json_handle, data='mstde', key=None, metric='auc', minmax = 'min'):
    '''
    json_handle : The Json Object for the experiment
    data : The data key to optimize over
    key : <List> key to factor the data into
    metric : Metric for comparison

    In this higher is better
    '''
    best_auc = dict()  # {} - np.inf
    best_params = dict()  # {}None
    best_run = dict()  # {}
    # get all the ke
    best_data = dict()
    if not isinstance(key, list):
        keys = json_handle[key]
        for k in keys:
            if minmax == 'min':
                best_auc[k] = + np.inf
            elif minmax == 'max':
                best_auc[k] = - np.inf
            else:
                raise Exception(f"minmax should be min or max, not {minmax}")
            best_params[k] = None
            best_run[k] = {}
            best_data[k] = dict()
        iterable = get_param_iterable_runs(json_handle)
        # iterable of all params
        # print(iterable)
        for i in iterable:
            folder, file = create_file_name(i, 'processed')
            filename = folder + file + '.pcsd'

            if not os.path.exists(filename):
                raise Exception(f"Processed Data  File Not Found, please process data first for : {filename}")
            # load the data
            data_obj = pkl_loader(filename)
            # data = torch.load(filename)

            mean = data_obj[data]['mean']
            stderr = data_obj[data]['stderr']
            # @dhawal
            if metric == 'auc':
                auc = np.mean(mean)
            else:
                auc = np.mean(mean[-mean.shape[0] // 10:])
                # auc = np.mean(mean[-200:])

            k = i[key]
            if minmax == 'min':
                if auc < best_auc[k]:
                    best_auc[k] = auc
                    best_run[k]['mean'] = mean
                    best_run[k]['stderr'] = stderr
                    best_params[k] = i
                    best_data[k] = data_obj
            elif minmax == 'max':
                if auc > best_auc[k]:
                    best_auc[k] = auc
                    best_run[k]['mean'] = mean
                    best_run[k]['stderr'] = stderr
                    best_params[k] = i
                    best_data[k] = data_obj

        return best_run, best_params, keys, best_data
    else:
        keys_list = []
        keys = []
        for k in key:
            keys_list.append(json_handle[k])
        iterators = itertools.product(*keys_list)
        for i in iterators:
            keys.append(i)
            if minmax == 'min':
                best_auc[i] = + np.inf
            elif minmax == 'max':
                best_auc[i] = - np.inf
            else:
                raise Exception(f"minmax should be min or max, not {minmax}")
            best_params[i] = None
            best_run[i] = dict()
            best_data[i] = dict()
        iterable = get_param_iterable_runs(json_handle)
        for i in iterable:
            folder, file = create_file_name(i, 'processed')
            filename = folder + file + '.pcsd'

            if not os.path.exists(filename):
                raise Exception(f"Processed Data  File Not Found, please process data first for : {filename}")
            # load the data
            data_obj = pkl_loader(filename)
            mean = data_obj[data]['mean']
            stderr = data_obj[data]['stderr']
            if metric == 'auc':
                auc = np.mean(mean)
            else:
                auc = np.mean(mean[-200:])
            val = []
            for k in key:
                val.append(i[k])
            index = tuple(val)
            if minmax == 'min':
                if auc < best_auc[index]:
                    best_auc[index] = auc
                    best_run[index]['mean'] = mean
                    best_run[index]['stderr'] = stderr
                    best_params[index] = i
                    best_data[index] = data_obj
            elif minmax == 'max':
                if auc > best_auc[index]:
                    best_auc[index] = auc
                    best_run[index]['mean'] = mean
                    best_run[index]['stderr'] = stderr
                    best_params[index] = i
                    best_data[index] = data_obj
                
        return best_run, best_params, keys, best_data


def get_key_from_dict(json_handle, key, key_key):
    list_ = json_handle[key]
    keys = []
    for l in list_:
        keys.append(l[key_key])
    return keys


# FIXME , fix this for valdation test
def find_best_key_key(json_handle, data='valid', key=None, key_key=None, metric='auc'):
    '''
    TODO incomplete for this repo
    Format of json handle : Should have list for all the parameters we want to search across
    key : If Key is none, it will find the best across all parameters else it will find the best across the key poarameter if key is list it will return teh best parameter for each configurations
    metric : auc or end (end :means the last 10 %  performance)
    '''
    best_auc = dict()  # {} - np.inf
    best_params = dict()  # {}None
    best_run = dict()  # {}
    # get all the ke
    best_data = dict()
    if not isinstance(key, list):
        keys = get_key_from_dict(json_handle, key, key_key)

        for k in keys:
            best_auc[k] = -np.inf
            best_params[k] = None
            best_run[k] = {}
            best_data[k] = dict()
        iterable = get_param_iterable_runs(json_handle)
        # iterable of all params
        # print(iterable)
        for i in iterable:
            folder, file = create_file_name(i, 'processed')
            filename = folder + file + '.pcsd'
            if not os.path.exists(filename):
                raise Exception(f"Processed Data  File Not Found, please process data first for : {filename}")
            # load the data
            data_obj = pkl_loader(filename)
            # data = torch.load(filename)

            mean = data_obj[data]['mean']
            stderr = data_obj[data]['stderr']
            auc = np.mean(mean)
            k = i[key][key_key]
            if auc > best_auc[k]:
                best_auc[k] = auc
                best_run[k]['mean'] = mean
                best_run[k]['stderr'] = stderr
                best_params[k] = i
                best_data[k] = data_obj

        return best_run, best_params, keys, best_data


def find_best_key_subkeys(json_handle, subkeys: list, data='valid', key=None, metric='auc'):
    '''
    '''
    # make individual dictinary for all the subkeys

    best_auc = dict()  # {} - np.inf
    best_params = dict()  # {}None
    best_run = dict()  # {}
    # get all the ke
    best_data = dict()
    if not isinstance(key, list):
        keys_list = []
        sub_keys = []
        for k in subkeys:
            keys_list.append(json_handle[k])
        iterators = itertools.product(*keys_list)
        keys = json_handle[key]
        for i in iterators:
            sub_keys.append(i)
            best_auc[i] = dict()
            best_params[i] = dict()
            best_run[i] = dict()
            best_data[i] = dict()
            for k in keys:
                best_auc[i][k] = -np.inf
                best_params[i][k] = None
                best_run[i][k] = {}
                best_data[i][k] = dict()
        iterable = get_param_iterable_runs(json_handle)

        for i in iterable:
            folder, file = create_file_name(i, 'processed')
            filename = folder + file + '.pcsd'

            if not os.path.exists(filename):
                raise Exception(f"Processed Data  File Not Found, please process data first for : {filename}")
            # load the data
            data_obj = pkl_loader(filename)
            # data = torch.load(filename)

            mean = data_obj[data]['mean']
            stderr = data_obj[data]['stderr']
            auc = np.mean(mean)
            val = []
            for k in subkeys:
                val.append(i[k])
            index = tuple(val)
            k = i[key]
            if auc > best_auc[index][k]:
                best_auc[index][k] = auc
                best_run[index][k]['mean'] = mean
                best_run[index][k]['stderr'] = stderr
                best_params[index][k] = i
                best_data[index][k] = data_obj

        return best_run, best_params, keys, sub_keys, best_data
    else:
        raise NotImplementedError
