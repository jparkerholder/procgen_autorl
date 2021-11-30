

#import ray.rllib.agents.dqn as dqn
import numpy as np
import argparse
import torch

from search_space import get_hparams

def convert_to_vec(args, config):
    df_hparams = get_hparams(args)
    using = []

    if args.experiment == "data_aug":
        for i in range(len(df_hparams)):
            row = df_hparams.iloc[i]
            using.append(config[row.Name])
    else:
        raise Exception("only implemented the data_aug experiment")

    return (using)


# converts a hparam df to a config
def convert_to_config(args, df_hparams):

    if args.search == 'cocabo':
        config = get_base_config(args)
        if args.experiment == "exploration":

            for i in range(len(df_hparams)):
                row = df_hparams.iloc[i]
                type_vec = row.Type
                if row.Name == "exploration_type":
                    if args.env.split('_')[0] == 'Atari':
                        if type_vec == 'continuous':
                            config['exploration_config'] = exp_atari[row.Use]
                        else:
                            config['exploration_config'] = exp_atari[row.Range[int(row.Use)]]
                    else:
                        if type_vec == 'continuous':
                            config['exploration_config'] = exp_basic[row.Use]
                        else:
                            config['exploration_config'] = exp_basic[row.Range[int(row.Use)]]
                else:
                    if type_vec == 'continuous':
                        config[row.Name] = row.Use
                    else:

                        config[row.Name] = row.Range[row.Use]

        elif args.experiment == 'data_aug':
            for i in range(len(df_hparams)):
                row = df_hparams.iloc[i]
                type_vec = row.Type
                if type_vec == 'continuous':
                    config[row.Name] = row.Use
                else:
                    config[row.Name] = row.Range[int(row.Use)]
        else:
            raise Exception("only implemented exploration and data_aug so far")

        return (config)

    else:
        config = get_base_config(args)
        if args.experiment == "exploration":

            for i in range(len(df_hparams)):
                row = df_hparams.iloc[i]
                if row.Name =="exploration_type":
                    if args.env.split('_')[0] == 'Atari':
                        config['exploration_config'] = exp_atari[row.Use]
                    else:
                        config['exploration_config'] = exp_basic[row.Use]
                else:
                    config[row.Name] = row.Use

        elif args.experiment == 'data_aug':
            for i in range(len(df_hparams)):
                row = df_hparams.iloc[i]
                config[row.Name] = row.Use
        else:
            raise Exception("only implemented exploration and data_aug so far")

        return(config)

# samples from each hparam range, generates a config with it
def get_random_config(args):
    
    df_hparams = get_hparams(args)
    
    to_use = []
    for i in range(len(df_hparams)):
        row = df_hparams.iloc[i]
        if row.Type == 'continuous':
            to_use.append(np.random.uniform(row.Range[0], row.Range[1]))
        elif row.Type == 'categorical':
            if args.cat_exp == 'fixed':
                if args.search == 'cocabo':
                    to_use.append(0)
                else:
                    to_use.append(args.fixed_cat_val)
            else:
                if args.search == 'cocabo':
                    to_use.append(round(np.random.uniform()*(len(row.Range)-1)))
                else:
                    to_use.append(row.Range[round(np.random.uniform()*(len(row.Range)-1))])
        else:
            raise Exception('Must be either continuous or categorical')
    df_hparams['Use'] = to_use
    config = convert_to_config(args, df_hparams)
    return(config)

def get_base_config(args):
    
    if args.experiment == "exploration":
        
        if args.algo == 'dqn':
            config = dqn.DEFAULT_CONFIG.copy()
            
        config["num_gpus"] = args.gpu_per_trial
        config["num_workers"] = args.cpu_per_trial
        
        if args.env == 'frozenlake':
            config["env"] = "FrozenLake-v0"
            config["env_config"] = {
                        "desc": [
                            "SFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFF",
                            "FFFFFFFFFFFFFFFG",
                        ],
                        "is_slippery": False
            }
            # Limit horizon to make it really hard for non-curious agent to reach
            # the goal state.
            config["horizon"] = 40
            
        elif args.env.split('_')[0] == 'Atari':
            
            config["env"] = '{}NoFrameskip-v4'.format(args.env.split('_')[1])
            config['hiddens'] = [512]
            config['double_q'] = False
            config['dueling'] = False
            config['prioritized_replay'] = False
            config['num_atoms'] = 1
            config['target_network_update_freq'] = 8000
            config['lr'] = 0.0000625
            config['adam_epsilon'] = 0.00015
            config['buffer_size'] = 1000000
            config['rollout_fragment_length'] = 4
            config['train_batch_size'] = 32
            config['timesteps_per_iteration'] = 10000
            config['learning_starts'] = 20000
            config['prioritized_replay_alpha']: 0.5
            config['final_prioritized_replay_beta']: 1.0
            config['prioritized_replay_beta_annealing_timesteps'] = 2000000
            config['n_step'] = 1
            #config['model']['grayscale'] = True
            
    elif args.experiment == 'data_aug':
        
        if len(args.env.split('-')) >1:
            ## DMC
            
            # PPO Arguments. 
            config = {}
            config['lr'] = 7e-4
            config['eps'] = 1e-5 # RMSprop epsilon
            config['alpha'] = 0.99 # RMSProp alpha
            config['gamma']=0.99 # discount factor for rewards
            config['gae_lambda'] = 0.95 # gae lambda parameter
            config['entropy_coef'] = 0.01 # entropy term coefficient
            config['value_loss_coef'] = 0.5 #value loss coefficient (default: 0.5)
            config['max_grad_norm'] = 0.5 #max norm of gradients
            config['seed']=1
            config['num_processes'] = 4 # how many training CPU processes to use
            config['num_steps'] = 500 # number of forward steps in A2C
            config['ppo_epoch'] = 3 # number of ppo epochs
            #config['num_mini_batch']=32 # number of batches for ppo
            config['clip_param'] = 0.2 # ppo clip parameter
            config['log_interval'] = 1 # log interval, one log per n updates
            config['save_interval'] = 1 # save interval, one save per n update
            config['num_env_steps'] = 10e6 # number of environment steps to train
            config['hidden_size'] = 512 # state embedding dimension
            config['train_resource_files'] = 'auto-drac-dmc/distractors/images/*mp4'
            config['frame_stack'] = 3
            config['total_frames'] = 1000
            # DrAC Arguments.
            config['use_sacae_network']=False
            config['aug_type']='crop' # augmentation type
            config['aug_coef'] = 0.1 # coefficient on the augmented loss
            config['aug_extra_shape'] = 0 # increase image size by
            config['image_pad'] = 12 # increase image size by
            config['preempt'] = False # safe preemption: load the latest checkpoint with same args and continue training)
            config['cuda'] = True if args.gpu_per_trial >0 else False
        else:
            ## ProcGen
            
            # PPO Arguments. 
            config = {}
            config['lr'] = 5e-4
            config['eps'] = 1e-5 # RMSprop epsilon
            config['alpha'] = 0.99 # RMSProp alpha
            config['gamma']=0.999 # discount factor for rewards
            config['gae_lambda'] = 0.95 # gae lambda parameter
            config['entropy_coef'] = 0.01 # entropy term coefficient
            config['value_loss_coef'] = 0.5 #value loss coefficient (default: 0.5)
            config['max_grad_norm'] = 0.5 #max norm of gradients
            config['seed']=1
            config['num_processes'] = 64 # how many training CPU processes to use
            config['num_steps'] = 256 # number of forward steps in A2C
            config['ppo_epoch'] = 3 # number of ppo epochs
            config['num_mini_batch']=8 # number of batches for ppo
            config['clip_param'] = 0.2 # ppo clip parameter
            config['log_interval'] = 1 # log interval, one log per n updates
            config['save_interval'] = 1 # save interval, one save per n update
            config['num_env_steps'] = 25e6 # number of environment steps to train
            config['hidden_size'] = 256 # state embedding dimension
            # Procgen Arguments.
            config['distribution_mode']='easy' # distribution of envs for procgen
            config['num_levels'] = 200 # number of Procgen levels to use for training
            config['start_level'] = 0 # start level id for sampling Procgen levels
            # DrAC Arguments.
            config['aug_type']='crop' # augmentation type
            config['aug_coef'] = 0.1 # coefficient on the augmented loss
            config['aug_extra_shape'] = 0 # increase image size by
            config['image_pad'] = 12 # increase image size by
            config['preempt'] = False # safe preemption: load the latest checkpoint with same args and continue training)
            config['cuda'] = True if args.gpu_per_trial >0 else False
        
    else:
        raise Exception("Only works for exploration and data_aug")
        
    return(config)