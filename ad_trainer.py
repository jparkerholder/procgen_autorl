
import copy
import os
import time
from collections import deque

import numpy as np
import torch
import time

from utils import get_base_config

from auto_drac.ucb_rl2_meta import algo, utils
from auto_drac.ucb_rl2_meta.model import Policy, AugCNN
from auto_drac.ucb_rl2_meta.storage import RolloutStorage

from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from auto_drac.ucb_rl2_meta.envs import VecPyTorchProcgen, TransposeImageProcgen

from auto_drac import data_augs

from search_space import aug_to_func

class ADA_Trainer(object):
    
    def __init__(self, args, config, agent):
        
        self.config = config
        self.env_name = args.env
        self.device ="cuda:0" if args.gpu_per_trial >0 else "cpu"
        self.criteria = args.budget_type
        self.t_ready = args.t_ready
        self.seed = args.seed + agent * 10000
        self.config['seed'] = self.seed
        
        venv = ProcgenEnv(num_envs=self.config['num_processes'], env_name=self.env_name, \
        num_levels=self.config['num_levels'], start_level=self.config['start_level'], \
        distribution_mode=self.config['distribution_mode'])
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False)
        self.envs = VecPyTorchProcgen(venv, self.device)
        self.obs_shape = self.envs.observation_space.shape
        self.actor_critic = Policy(
            self.obs_shape,
            self.envs.action_space.n,
            base_kwargs={'recurrent': False, 'hidden_size': self.config['hidden_size']})        
        self.actor_critic.to(self.device)
        
        batch_size = int(self.config['num_processes'] * self.config['num_steps'] / self.config['num_mini_batch'])
    
        self.aug_id = data_augs.Identity
        self.aug_func = aug_to_func[self.config['aug_type']](batch_size=batch_size)
        
        self.agent = algo.DrAC(
                self.actor_critic,
                self.config['clip_param'],
                self.config['ppo_epoch'],
                self.config['num_mini_batch'],
                self.config['value_loss_coef'],
                self.config['entropy_coef'],
                lr=self.config['lr'],
                eps=self.config['eps'],
                max_grad_norm=self.config['max_grad_norm'],
                aug_id=self.aug_id,
                aug_func=self.aug_func,
                aug_coef=self.config['aug_coef'],
                env_name=self.env_name)
        
    def restore(self, checkpoint_path):
                
        if os.path.exists(checkpoint_path):
            print("restoring from path")
            if self.device == "cpu":
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(checkpoint_path)
            self.agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            pass
    
    def save(self, checkpoint_dir):
        
        torch.save({
                        'model_state_dict': self.agent.actor_critic.state_dict(),
                        'optimizer_state_dict': self.agent.optimizer.state_dict(),
                }, os.path.join(checkpoint_dir)) 
        
    def train(self):
        
        rollouts = RolloutStorage(self.config['num_steps'], 
                                   self.config['num_processes'],
                                   self.envs.observation_space.shape, 
                                   self.envs.action_space,
                                   self.actor_critic.recurrent_hidden_state_size,
                                   aug_type=self.config['aug_type'], 
                                   split_ratio=0.1)

            
        obs = self.envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(self.device)
    
        episode_rewards = deque(maxlen=100)
            
        time_0 = time.time()
        ts = 0
        
        finished = False
    
        while not finished:
            self.actor_critic.train()
            for step in range(self.config['num_steps']):
                # Sample actions
                with torch.no_grad():
                    obs_id = self.aug_id(rollouts.obs[step])
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        obs_id, rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
    
                # Obser reward and next obs
                obs, reward, done, infos = self.envs.step(action)
                
                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
    
                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
    
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)
    
            with torch.no_grad():
                obs_id = self.aug_id(rollouts.obs[-1])
                next_value = self.actor_critic.get_value(
                    obs_id, rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()
                
            rollouts.compute_returns(next_value, self.config['gamma'], self.config['gae_lambda'])
    
            value_loss, action_loss, dist_entropy = self.agent.update(rollouts)    
            rollouts.after_update()
    
            # save for every interval-th episode or for the last epoch
            ts += self.config['num_processes'] * self.config['num_steps']
            #print("\nStep {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}"
            #        .format(ts, len(episode_rewards), np.mean(episode_rewards),
            #                np.median(episode_rewards)))
                
    
            ### Eval on the Full Distribution of Levels ###
            eval_episode_rewards = self.test()
            
            mean_test = np.mean(eval_episode_rewards)
            median_test = np.median(eval_episode_rewards)
            
            time_elapsed = time.time() - time_0
            if self.criteria == 'timesteps_total':
                if ts >= self.t_ready:
                    finished = True
            elif self.criteria == 'time_total_s':
                if time_elapsed >= self.t_ready:
                    finished = True
        result = {}
        result['episode_reward_mean'] = np.mean(episode_rewards)
        result['timesteps_total'] = ts
        result['time_total_s'] = time_elapsed
        result['test_episode_reward_mean'] = mean_test
        return(result)
            
    def test(self, num_processes=1, num_evals=100, num_levels=0):
        
        self.actor_critic.eval()
    
        # Sample Levels From the Full Distribution 
        venv = ProcgenEnv(num_envs=num_processes, env_name=self.env_name, \
        num_levels=num_levels, start_level=0, \
        distribution_mode=self.config['distribution_mode'])
        
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False)
        eval_envs = VecPyTorchProcgen(venv, self.device)

        eval_episode_rewards = []

        obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(
            num_processes, self.actor_critic.recurrent_hidden_state_size, device=self.device)
        eval_masks = torch.ones(num_processes, 1, device=self.device)

        while len(eval_episode_rewards) < num_evals:
            with torch.no_grad():
                obs_id = self.aug_id(obs)
                value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        obs_id, eval_recurrent_hidden_states,
                        eval_masks, deterministic=False)

            obs, _, done, infos = eval_envs.step(action)
             
            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=self.device)
    
            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])
    
        eval_envs.close()
    
        #print("Last {} test episodes: mean/median reward {:.1f}/{:.1f}\n"\
        #    .format(len(eval_episode_rewards), \
        #    np.mean(eval_episode_rewards), np.median(eval_episode_rewards)))
    
        return eval_episode_rewards
        
            
            
