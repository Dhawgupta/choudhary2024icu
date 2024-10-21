# The ICU-Sepsis Environment (Baseline Algorithms Implementation)

The **ICU-Sepsis** environment is a reinforcement learning environment that
simulates the treatment of sepsis in an intensive care unit (ICU). The
environment is introduced in the paper
[ICU-Sepsis: A Benchmark MDP Built from Real Medical Data](https://arxiv.org/abs/2406.05646),
accepted at the Reinforcement Learning Conference, 2024. ICU-Sepsis is built
using the [MIMIC-III Dataset](https://physionet.org/content/mimiciii/1.4/),
based on the work of
[Komorowski et al. (2018)](https://www.nature.com/articles/s41591-018-0213-5). The environment can be found at the following [Repository](https://github.com/icu-sepsis/icu-sepsis/tree/main).


Citation:
```bibtex
@inproceedings{
  choudhary2024icusepsis,
  title={{ICU-Sepsis}: A Benchmark {MDP} Built from Real Medical Data},
  author={Kartik Choudhary and Dhawal Gupta and Philip S. Thomas},
  booktitle={Reinforcement Learning Conference},
  year={2024},
  url={https://arxiv.org/abs/2406.05646}
}
```
## Code Organization

The code is organized as follows:

`experiments`: Contains the JSON files containing the hyperparameters and configurations for the different methods. 

`src`: Contains the source code for different algorithms. 

`run`: Contains the scripts to run the experiments.

### Running the code using json file. 

For example an experiment specified in `experiments/debug.json` can be run using the following command. 

```bash
python src/mainjson.py experiments/debug.json 0
```
The above command will run the first configuration in the `debug.json` file.


### Executing a sweep
Using GNU parallel we can run multiple configurations in parallel. 

```bash
python run/local.py -p src/mainjson.py -j experiments/debug.json
``` 
The above command will try to run all possible configurations in parallel. One can limit the number of threads based on their system using `-c` flag. 

The results for the experiments are stored in `results` folder. Each file in the results folder stores a single hyperparameter configuration with a single seed value. 

#### Process results
We then need to aggregate the results across different seeds using the following command. 

```bash
python analysis/process_data.py experiments/debug.json
```

#### Plot the results
Finally, we can plot the results using the following command. 

```bash
python analysis/learning_curve.py y returns auc experiments/debug.json
```
The above plots the returns using area under the curve (AUC) as a metric to select the best parameters, for the configurations in the `debug.json` file.

The plot is created in the `plots` folder.
