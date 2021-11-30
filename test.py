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
#import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print
import copy
import torch
import gym
import imageio


from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from auto_drac.ucb_rl2_meta.envs import VecPyTorchProcgen, TransposeImageProcgen

from pbt import PBT
from ad_trainer import ADA_Trainer
from ad_trainer_dmc import ADA_Trainer_DMC
from utils import get_random_config, convert_to_vec

def get_search_master(args):
        
    if args.search in ['PBT', 'PB2']:
        return(PBT(args))

def test(args):
    
    args.fixed = False  
    args.cat_exp = "Random"
    args.search = "PB2"
    args.budget_type = "timesteps_total"
    args.t_ready = 500000
    config = get_random_config(args)
    
    train_perf = []
    test_perf = []

    for agent in range(args.batchsize):
            
        print("\nTesting Agent: {}\n".format(agent))
            
        checkpoint_path = args.agent_filepath + '_seed' + str(args.seed) + '_Agent' + str(agent)

        if len(args.env.split('-'))>1:
            trainer = ADA_Trainer_DMC(args, config, agent)
        else:
            trainer = ADA_Trainer(args, config, agent)
            checkpoint_path += ".pt"

                
            # print(dir(trainer))
            trainer.restore('../earl_checkpoints/' + checkpoint_path)
                
            #train = trainer.test(num_evals = 100, num_levels = config['num_levels'])
            #test = trainer.test(num_evals = 100, num_levels = 0)

            #print("\nAgent: {}, Train: {}, Test: {} \n".format(agent, np.mean(train), np.mean(test)))
            
            #train_perf.append(np.mean(train))
            #test_perf.append(np.mean(test))
    
    if args.record:
        make_fn = lambda x: gym.make('procgen:procgen-{}-v0'.format(args.env))
        trainer.actor_critic.eval()
    
        # Sample Levels From the Full Distribution 
        venv = ProcgenEnv(num_envs=1, env_name=trainer.env_name, \
        num_levels=1, start_level=0, \
        distribution_mode=trainer.config['distribution_mode'])
        
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False)
        eval_envs = VecPyTorchProcgen(venv, trainer.device)

        eval_episode_rewards = []

        obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(
            1, trainer.actor_critic.recurrent_hidden_state_size, device=trainer.device)
        eval_masks = torch.ones(1, 1, device=trainer.device)
        
        images = []
        
        img = eval_envs.render(mode='rgb_array')
        
        while len(eval_episode_rewards) < 1:
            images.append(img)
            with torch.no_grad():
                obs_id = trainer.aug_id(obs)
                value, action, action_log_prob, recurrent_hidden_states = trainer.actor_critic.act(
                        obs_id, eval_recurrent_hidden_states,
                        eval_masks, deterministic=False)

            obs, _, done, infos = eval_envs.step(action)
            img = eval_envs.render(mode='rgb_array')
             
            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=trainer.device)
    
            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])
    
        eval_envs.close()
        imageio.mimsave('l{}.gif'.format(args.env), [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
        
    else:
        df_test = pd.DataFrame({'Seed': [args.seed for _ in range(args.batchsize)], 'Train': train_perf, 'Test': test_perf})
        df_test.to_csv('../earl_data/test/{}_seed{}.csv'.format(args.agent_filepath, args.seed))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='starpilot')  ## for atari put Atari_X for game X
    parser.add_argument('--experiment',  type=str, default='data_aug')  ## ['exploration', 'data_aug']
    parser.add_argument('--algo', type=str, default='dqn')  ## this is only used in the exploration experiment
    parser.add_argument('--batchsize', '-b', type=int, default=4)  ## Greater than 1 = PBT/PB2/New
    parser.add_argument('--df_filepath', type=str, default='df_all_starpilot_PB2_4Agents_cocabo_all_seed0')
    parser.add_argument('--agent_filepath', type=str, default='starpilot_PB2_cocabo')
    parser.add_argument('--seed', '-s', type=int, default=101)  ## Greater than 1 = PBT/PB2/New
    parser.add_argument('--gpu_per_trial', '-gp', type=int, default=0)  ## Greater than 1 = PBT/PB2/New
    parser.add_argument('--record', '-re', type=int, default=1)  ## Make video 1/0
    
   
    # PBT meta parameters here. E.g. T_ready, % of top/bottom to replace.

    args = parser.parse_args()

    if not os.path.exists('../earl_checkpoints/'):
        os.mkdir('../earl_checkpoints/')

    if not os.path.exists('../earl_data'):
        os.mkdir('../earl_data')

    if not os.path.exists('../earl_data/test'):
        os.mkdir('../earl_data/test')

    test(args)


if __name__ == '__main__':
    main()
