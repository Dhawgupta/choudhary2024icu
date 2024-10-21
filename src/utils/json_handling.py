'''
This codeabase will help us handle the json reading and stuff
'''
import collections
import json
import itertools

def get_sorted_dict(json_name):
    with open(json_name, 'r') as fil:
        d = json.load(fil) 
    d = collections.OrderedDict(sorted(d.items()))
    return d

def get_sorted_dict_loaded(loaded_json_name):
    d = collections.OrderedDict(sorted(loaded_json_name.items()))
    return d


def get_param_iterable(d):
    '''
    Takes in a dictionary and makes an iterable vector out of that
    which contains all teh parameters once
    '''
    d_lists  = {}
    list_keys = []
    lists = []
    lists_non_keys = []
    for k in d.keys():
        if isinstance( d[k] , list):
            d_lists[k] = d[k]
            list_keys.append(k)
            lists.append(d[k])
        else:
            lists_non_keys.append(k)
     
    iterators = itertools.product(*lists)
    all_parameters = []
    for it in iterators:
        temp = dict()
        for i,k in enumerate(list_keys):
            temp[k] =  it[i]
        for k in lists_non_keys:
            temp[k] = d[k]
        all_parameters.append(temp)
    return all_parameters


def get_param_iterable_runs(d):
    '''
    Takes in a dictionary and makes an iterable vector out of that
    which contains all teh parameters once , with the list of all seeds
    Get the file name witout all the avereging quantities.
    '''
    d_lists = {}
    list_keys = []
    lists = []
    lists_non_keys = []
    for k in d.keys():
        if isinstance(d[k], list):
            if k == 'seed' : # average over the seeds and fold_no
                lists_non_keys.append(k)
                continue
            d_lists[k] = d[k]
            list_keys.append(k)
            lists.append(d[k]) # append all the parameters
        else:
            lists_non_keys.append(k)

    iterators = itertools.product(*lists) # iterate over all configurations of parameters
    all_parameters = []
    for it in iterators:
        temp = dict()
        for i, k in enumerate(list_keys):
            temp[k] = it[i]
        for k in lists_non_keys:
            temp[k] = d[k]
        all_parameters.append(temp)
    return all_parameters # returns all the parameters config, with things to average over 
