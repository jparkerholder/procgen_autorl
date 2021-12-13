# Population Based Training for OpenAI Procgen

This repo contains a variety of PBT style algorithms for tuning the hyperparameters of an actor-critic agent on the Procgen benchmark.

Please note that this code is not optimized for speed, due to the compute resources we had available was just a single GPU. We instead run the agents sequentially, and update the population in a synchronous fashion once each agent has reached the next checkpoint. This is a limitation of the implementation rather than the methods - to run PBT/PB2 with parallel resources please see the [ray library](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pb2_ppo_example.py).

## Installation

We include a requirements.txt file with the main open source libraries:
```
pip install -r requirements.txt
```

We include the baselines repo used for our RL agents, which can be installed as follows:
```
cd auto_drac
cd baselines
pip install -e .
cd ..
cd ..
```

## Code Format

- train.py: the main wrapper. Creates a dictionary of agents. initialize randomly. Then explore at the end of every (synchronous) update. This can be extended to be asynch if needed.
- pbt.py: contains PBT explore exploit.
- search_space.py: this is the parameter ranges. 
- utils.py: this contains the base config.
- auto_drac: this contains code from [the DrAC paper.](https://github.com/rraileanu/auto-drac)
- exp3_multiple_play: this contains the code for TV.EXP3.M.


## Running the Experiments

To run the Procgen experiments in the paper, use the following commands:

PBT:
```
python train.py --env bigfish --search PBT --cat_exp random --seed 100 --gpu_per_trial 1 --max_budget 25000000 --t_ready 500000 --batchsize 4  
```

PB2-Rand:
```
python train.py --env bigfish --search PB2 --cat_exp random --seed 100 --gpu_per_trial 1 --max_budget 25000000 --t_ready 500000 --batchsize 4  
```

PB2-Mult:
```
python train.py --env bigfish --search PB2 --cat_exp exp3_dep --seed 100 --gpu_per_trial 1 --max_budget 25000000 --t_ready 500000 --batchsize 4  
```

PB2-Mix:
```
python train.py --env bigfish --search PB2 --cat_exp cocabo --seed 100 --gpu_per_trial 1 --max_budget 25000000 --t_ready 500000 --batchsize 4  
```

## Synthetic example

There is also a toy problem here which may be interesting for rapidly developing new PBT algorithms: 

[Notebook](https://colab.research.google.com/drive/1m_DFF_OCSOGO14th4pdbOdAtkmCoWSkn?usp=sharing)

## Citation

If you found this useful, please cite the following:

```
@inproceedings{parkerholder2021tuning,
  author    = {Jack Parker{-}Holder and
               Vu Nguyen and
               Shaan Desai and
               Stephen J. Roberts},
  title     = {Tuning Mixed Input Hyperparameters on the Fly for Efficient Population
               Based AutoRL},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {34},
  year      = {2021}
}
```

## Acknowledgements

The base RL agent comes from the following [open source repo](https://github.com/rraileanu/auto-drac).

If you use this agent for your research, please consider also citing the associated paper.
