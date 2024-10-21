# make this into a file which uses argparse to run experiments
import os, sys, time
sys.path.append(os.getcwd())
from src.run_experiment import run_experiment
from src.utils.json_handling import get_sorted_dict, get_param_iterable



    # make the objects

if __name__ == '__main__':
    # get the parameters
    if len(sys.argv) < 3:
        print("Usage: python main.py <json_file> <idx>")
        exit()
    json_file = sys.argv[1]
    idx = int(sys.argv[2])

    d = get_sorted_dict(json_file)
    experiments = get_param_iterable(d)
    experiment = experiments[idx%len(experiments)]
    run_experiment(experiment, False)
    