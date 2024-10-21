import os, sys
from src.utils.formatting import create_file_name , get_folder_name

def check_experiment_not_done(experiment, list_of_done_experiments = None):
    '''
    Returns True if experiment is yet to be done
    '''
    # remove these
    experiment.pop('use_tensorboard', None)
    experiment.pop('track', None)

    folder, file = create_file_name(experiment)
    # print(file)
    file_name_check = folder + file + '.dw'
    if list_of_done_experiments is not None:
        if file + '.dw' not in list_of_done_experiments:
            return True
        return False
    # check ifn th efile exists
    if not os.path.exists(file_name_check):
        return True
    return False

def get_list_pending_experiments(experiments):
    '''
    Inputs : <list> of expeirments
    Returns : Index of pending experiments
    '''
    # given a list of expeiments
    pending_experiments = []
    experiment_no = len(experiments)
    # get folder name and the experiments in those
    foldername = get_folder_name(experiments[0])
    # load all files

    print(foldername)
    # import time
    list_of_done_experiments = None
    if os.path.exists(foldername):
        list_of_done_experiments = os.listdir(foldername)
    # print(list_of_done_experiments)
    # time.sleep(10)

    for idx, exp in enumerate(experiments):
        print(f'Checking [{idx}/{experiment_no}]\r' , end = "")
        if check_experiment_not_done(exp, list_of_done_experiments):
            pending_experiments.append(idx)
    return pending_experiments

def get_list_pending_experiments_not_compeletelydebug(experiments):
    '''
    Inputs : <list> of expeirments
    Returns : Index of pending experiments
    '''
    # given a list of expeiments
    pending_experiments = []
    experiment_no = len(experiments)
    # get folder name and the experiments in those
    all_folder_names  = []
    for e in experiments:
        fldname = get_folder_name(e)
        if fldname not in all_folder_names:
            all_folder_names.append(fldname)
    
    list_of_done_experiments = []
    for fl in all_folder_names:
        print(fl)
        if os.path.exists(fl):
            list_of_done_experiments_ = os.listdir(fl)
            for l in list_of_done_experiments_:
                list_of_done_experiments.append(l)
   
    for idx, exp in enumerate(experiments):
        print(f'Checking [{idx}/{experiment_no}]\r' , end = "")
        if check_experiment_not_done(exp, list_of_done_experiments):
            pending_experiments.append(idx)
    return pending_experiments