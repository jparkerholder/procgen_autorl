import argparse
import os

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
import shutil
import ray
from ray import tune
import os
from ray.tune.logger import pretty_print
import copy

from pbt import PBT
from ad_trainer import ADA_Trainer
from utils import get_random_config, convert_to_vec

def get_search_master(args):
        
    if args.search in ['PBT', 'PB2']:
        return(PBT(args))
    
def restore(args):
    
    word = 'fixed' if args.fixed else 'adaptive'
    word2 = 'all' if  args.cat_exp is not 'fixed' else args.fixed_cat_val
    
    df_all = pd.read_csv("../pb2_data/df_all_{}_{}_{}Agents_{}_{}_seed{}.csv".format(args.env, args.search, args.batchsize, word, word2, args.seed))
    
    all_done = True if df_all.timesteps_total.max() > args.max_budget else False
    
    return df_all

def train(args):
    ## set up population, with fixed base config

    pop = {i: {'done': False, 'config': get_random_config(args), 'path': None} for i in range(args.batchsize)}
        
    ## set up exploration master
    master = get_search_master(args)

    if args.restore:
        df_all, all_done = restore(args)
    else:
        df_all = pd.DataFrame(columns=['Agent', 't', args.budget_type, 'R', 'R_test', 'conf', 'path'])
        all_done = False  # np.array([pop[agent]['done'] for agent in pop.keys()]).all()

    while not all_done:

        for agent in pop.keys():
            
            print("\nTraining Agent: {}\n".format(agent))
            
            checkpoint_path = '{}_{}_{}_seed{}_Agent{}'.format(args.env, args.search, args.cat_exp, args.seed, str(agent))

            if args.experiment == 'data_aug':
                trainer = ADA_Trainer(args, pop[agent]['config'], agent)
                checkpoint_path += ".pt"
            else:
                trainer = None
                
            # print(dir(trainer))
            if df_all[df_all['Agent'] == agent].t.empty:
                t = 1
            else:
                trainer.restore('../pb2_checkpoints/' + checkpoint_path)
                t = df_all[df_all['Agent'] == agent].t.max() + 1

            result = {}
            result[args.budget_type] = 0

            if args.debug:
                ## this is just for testing the explore step can figure out trivial settings
                while result[args.budget_type] < 1:
                    if pop[agent]['config']['aug_type'] == 'cutout':
                        result['episode_reward_mean'] = 100 * t/pop[agent]['config']['lr']
                    else:
                        result['episode_reward_mean'] = np.random.rand()  # *t
                    result['timesteps_total'] = t * 10000
                    result['test_episode_reward_mean'] = np.random.rand() #*t
            else:
                while result[args.budget_type] < args.t_ready:
                    result = trainer.train()


            reward = result['episode_reward_mean']
            if df_all[df_all['Agent'] == agent].empty:
                scalar_steps = result[args.budget_type]
            else:
                scalar_steps = result[args.budget_type] + df_all[df_all['Agent'] == agent][args.budget_type].max()

            print("\nAgent: {}, Timesteps: {}, Reward: {}\n".format(agent, scalar_steps, result['episode_reward_mean']))

            trainer.save('../pb2_checkpoints/' + checkpoint_path)
            pop[agent]['path'] = checkpoint_path
            reward_test = result['test_episode_reward_mean']

            conf = convert_to_vec(args, pop[agent]['config'])

            d = pd.DataFrame(columns=df_all.columns)
            d.loc[0] = [agent, t, scalar_steps, reward, reward_test, conf, checkpoint_path]
            df_all = pd.concat([df_all, d]).reset_index(drop=True)

            if df_all[df_all['Agent'] == agent][args.budget_type].max() >= args.max_budget:
                pop[agent]['done'] = True


        # Now all agents have completed, we explore!
        if args.search in ['PBT', 'PB2']:
            for agent in pop.keys():
                old_conf = convert_to_vec(args, pop[agent]['config'])
                pop[agent], copied = master.exploit(args, agent, df_all, pop)
                # here we need to include a way to account for changes in the data.
                new_conf = convert_to_vec(args, pop[agent]['config'])
                if not new_conf == old_conf:
                    print("changing conf for agent: {}".format(agent))
                    new_row = df_all[(df_all['Agent']==copied) & (df_all['t'] == df_all.t.max())]
                    new_row['Agent'] = agent
                    new_row['path'] = pop[agent]['path']
                    print("new row conf old: ", new_row['conf'])
                    print("new row conf new: ", [new_conf])
                    new_row['conf'] = [new_conf]        
                    df_all = pd.concat([df_all, new_row]).reset_index(drop=True)
                    print("new config: ", new_conf)
    
        word = 'all' if  args.cat_exp is not 'fixed' else args.fixed_cat_val
        
        df_all.to_csv(
            "../pb2_data/df_all_{}_{}_{}Agents_{}_{}_seed{}.csv".format(args.env, args.search, args.batchsize, args.cat_exp, word, args.seed))
        all_done = np.array([pop[agent]['done'] for agent in pop.keys()]).all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='bigfish')  ## for atari put Atari_X for game X
    parser.add_argument('--experiment',  type=str, default='data_aug')  ## ['data_aug']
    parser.add_argument('--batchsize', '-b', type=int, default=4)  ## Greater than 1 = PBT/PB2/New
    parser.add_argument('--seed', '-sd', type=int, default=0)  ## Use 0,1,2,3,4,5,6,7,8,9
    parser.add_argument('--cpu_per_trial', '-cp', type=int, default=2)  ## We run this for each agent
    parser.add_argument('--gpu_per_trial', '-gp', type=int, default=0)  ## We run this for each agent
    parser.add_argument('--search', '-s', type=str, default='PB2')  ## ['Fixed', 'Random', 'PBT', 'PB2']
    parser.add_argument('--max_budget', '-mb', type=int, default=1000000)  ## train samples * frameskip
    parser.add_argument('--t_ready', '-tr', type=int, default=20000)  ## how many steps between explore/exploit
    parser.add_argument('--budget_type', '-bt', type=str, default='timesteps_total')  ## could also be wall-clock
    parser.add_argument('--pbt_thresh', type=str, default=0.25)
    parser.add_argument('--pbt_resample', type=float, default=0.25)
    parser.add_argument('--fixed', type=bool, default=False)
    parser.add_argument('--cat_exp', type=str, default='cocabo') # ['fixed', 'random', 'exp3_indep', 'exp3_dep', 'cocabo']
    parser.add_argument('--fixed_cat_val', type=str, default='crop')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--restore', type=bool, default=False)
    
    args = parser.parse_args()

    if not os.path.exists('../pb2_checkpoints/'):
        os.mkdir('../pb2_checkpoints/')

    if not os.path.exists('../pb2_data'):
        os.mkdir('../pb2_data')

    train(args)

if __name__ == '__main__':
    main()
