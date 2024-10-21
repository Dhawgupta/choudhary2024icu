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
# allocation no required
# parser.add_argument("--allocation" ,"-a", type = str, default = 'rrg')
# works only for upto 24 hours

parser.add_argument("--time", '-t', type = int, default = 3)
parser.add_argument("--cpus", '-c', type = int, default = 16)
parser.add_argument("--memory", '-m', type = int, default = 32)
parser.add_argument('--ntasks', '-n', type=int, default = 4) # number of tasks per CPU
# parser.add_argument("--export", '-e', type = str, default = 'exports3.dat')
parser.add_argument('--pythonfile','-p', type = str)
parser.add_argument('--json', '-j', type = str ,nargs='+', help='Json Files', required=True) # the json configuration
parser.add_argument('--overwrite','-o', type = bool,   help= "If True, the experiment will overwrite past experiment", default = False)
parser.add_argument('--days' , '-d', type = int, help = "Number of days to run the code", default = 0)



args = parser.parse_args()

json_files = args.json
for json_file in json_files:
    print(json_file)
    d = get_sorted_dict(json_file)
    experiments = get_param_iterable(d)
    if not args.overwrite:
        pending_experiments = get_list_pending_experiments(experiments)
    else:
        pending_experiments = list(range(len(experiments)))

    print(f"Experiments : {json_file} : {pending_experiments}")
    # by default use only one node
    # allocation_name = args.allocation + '-whitem'
    time_str = "{}-{}:59:00".format(str(args.days) , str(args.time - 1))
    cpus = args.cpus * args.ntasks
    memory = args.memory


    num_commands = len(pending_experiments)
    # get the number of nodes that we want
    nodes = num_commands // cpus
    if num_commands % cpus == 0:
        nodes = nodes
    else:
        nodes = nodes + 1


    pythoncommands = []
    for idx in pending_experiments:
        com = 'python ' + args.pythonfile
        # for k in c.keys():
        #     com += ' --{} {}'.format(k, c[k])
        com += f' {json_file} {idx}'
        com+= '\n'
        pythoncommands.append(com)

    command_nodes = [ [] for i in range(nodes)]
    # give each node command
    for i,c in enumerate(pythoncommands):
        command_nodes[i%nodes].append(c)

    foldername = './temp/parallel_scripts/'
    if not os.path.exists(foldername):
            os.makedirs(foldername, exist_ok=True)
    filename = [f'./temp/parallel_scripts/node_{i}_{str(args.cpus)}_{str(np.random.randint(0,100000))}.txt' for i in range(nodes) ]

    # write commands in files
    for i,f in enumerate(filename):
        fil = open(f,'w')
        fil.writelines(command_nodes[i])
        fil.close()

    parallel_commands = []
    for f in filename:
        command = 'parallel --verbose -P -0 :::: {}'.format(f)
        parallel_commands.append(command)
    # make slurm file

    foldername = './temp/slurm_scripts/'
    if not os.path.exists(foldername):
            os.makedirs(foldername, exist_ok=True)
    slurm_files = [ f'./temp/slurm_scripts/slurm{i}.sh' for i in range(nodes)]
    cwd = os.getcwd()
    for n in range(nodes):
        slurm_file = f"#!/bin/bash\n" \
                    f"#SBATCH -t {time_str}\n" \
                    f"#SBATCH -c {args.cpus}\n" \
                    f"#SBATCH --mem={memory}G\n" \
                    f"#SBATCH -p cpu-preempt,cpu\n" \
                    f"#SBATCH -o slurm-%j.out\n"\
                    f"source ~/.bashrc\n"\
                    f"module load parallel\n"\
                    f"export MKL_SERVICE_FORCE_INTEL=1\n"\
                    f"conda activate cv\n" \
                    f"cd {cwd}\n" \
                    f"{parallel_commands[n]}"
        sfile = open(slurm_files[n], 'w')
        sfile.write(slurm_file)
        sfile.close()
        os.system(f'sbatch {slurm_files[n]}')
        time.sleep(1)

# f"export PYTHONPATH={cwd}:$PYTHONPATH\n" \
# f"source ~/cv/bin/activate\n" \
# f'export PATH="/mnt/nfs/home/dgupta/miniconda3/bin:$PATH"\n'\
                    