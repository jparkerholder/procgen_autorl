import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from auto_drac_dmc.a2c_ppo_acktr import algo, utils
from auto_drac_dmc.a2c_ppo_acktr.arguments import get_args
from auto_drac_dmc.a2c_ppo_acktr.envs import make_vec_envs
from auto_drac_dmc.a2c_ppo_acktr.model import Policy
from auto_drac_dmc.a2c_ppo_acktr.storage import RolloutStorage
from auto_drac_dmc.evaluation import evaluate

from auto_drac_dmc import dmc2gym 
from auto_drac_dmc import data_augs

from baselines import logger

aug_to_func = {    
        'crop': data_augs.Crop,
        'random-conv': data_augs.RandomConv,
        'grayscale': data_augs.Grayscale,
        'flip': data_augs.Flip,
        'rotate': data_augs.Rotate,
        'cutout': data_augs.Cutout,
        'cutout-color': data_augs.CutoutColor,
        'color-jitter': data_augs.ColorJitter,
}

def main():
    args = get_args()

    if (args.domain_name == 'finger' and args.task_name == 'spin') or \
        (args.domain_name == 'walker' and args.task_name == 'walk'):
        args.action_repeat = 2
    elif (args.domain_name == 'cartpole' and args.task_name == 'swingup'):
        args.action_repeat = 8 
    else:
        args.action_repeat = 4

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    log_file = '-{}-{}-{}-{}-dmc{}-rad{}-drac{}-{}-ucb{}-uec{}-sacae{}-gae{}-ld{}-tl{}-hs{}-fs{}-ns{}-np{}-lr{}-ec{}-pe{}-nmb{}-g{}-l{}-s{}'\
        .format(args.run_name, args.domain_name, args.task_name, args.train_img_source, \
        args.use_pixel_dmc, args.use_rad, args.use_drac, args.aug_type, args.use_ucb, args.ucb_exploration_coef, \
        args.use_sacae_network, args.use_gae, args.use_linear_lr_decay, args.use_proper_time_limits, \
        args.hidden_size, args.frame_stack, \
        args.num_steps, args.num_processes, args.lr, args.entropy_coef, args.ppo_epoch, args.num_mini_batch, \
        args.gamma, args.gae_lambda, args.seed)
    print("\nLog File: ", log_file)
    logger.configure(dir=args.log_dir, format_strs=['csv', 'stdout'], log_suffix=log_file)

    log_dir = os.path.expanduser(os.path.join(args.log_dir, "train" + log_file))
    eval_log_dir = os.path.expanduser(os.path.join(args.log_dir, "test" + log_file))
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    print("\nTrain Log Dir: ", log_dir)
    print("\nTest Log Dir: ", eval_log_dir)

    if args.train_img_source == 'natural':
        args.train_img_source = 'video'
        args.train_resource_files = 'distractors/natural/*mp4'
    elif args.train_img_source == 'artificial':
        args.train_img_source = 'video'
        args.train_resource_files = 'distractors/artificial/*mp4'
    else:
        args.train_resource_files = None

    print("\ntrain backgrounds: ", args.train_resource_files)

    envs = make_vec_envs(args, args.seed, args.num_processes, 
        args.gamma, log_dir, device, False, args.train_img_source, 
        args.train_resource_files, 
        args.frame_stack)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        use_sacae_network=args.use_sacae_network,
        base_kwargs={'recurrent': args.recurrent_policy, 'hidden_size': args.hidden_size})
    actor_critic.to(device)
    print(actor_critic)

    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)

    aug_id = data_augs.Identity
    if args.use_ucb:
        aug_list = [aug_to_func[t](batch_size=batch_size) 
            for t in list(aug_to_func.keys())]

        agent = algo.UCBDrAC(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            aug_list=aug_list,
            aug_id=aug_id,
            aug_coef=args.aug_coef,
            num_aug_types=len(list(aug_to_func.keys())),
            ucb_exploration_coef=args.ucb_exploration_coef,
            ucb_window_length=args.ucb_window_length)

    elif args.use_drac:
        aug_func = aug_to_func[args.aug_type](batch_size=batch_size)

        agent = algo.DrAC(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            aug_id=aug_id,
            aug_func=aug_func,
            aug_coef=args.aug_coef,
            env_name=args.env_name)

    elif args.use_rad:
        aug_func = aug_to_func[args.aug_type](batch_size=batch_size)

        agent = algo.RAD(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            aug_func=aug_func,
            aug_id=aug_id, 
            aug_prob=args.aug_prob)

    else:
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes // args.action_repeat
    print("\nNum Env Steps {}, Num Policy Steps {}, Num Updates {}"\
        .format(args.num_env_steps, int(args.num_env_steps / args.action_repeat), num_updates))
    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

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
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        if args.use_ucb and j > 0:
            agent.update_ucb_values(rollouts)
        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # Save Model
        if (j > 0 and j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.run_name)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, "agent-{}".format(j) + log_file + ".pt")) 

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            env_num_steps = int(args.action_repeat * total_num_steps)
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

            logger.logkv("train/nupdates", j)
            logger.logkv("train/total_num_steps", total_num_steps)            

            logger.logkv("train/num_policy_steps", total_num_steps)            
            logger.logkv("train/num_env_steps", env_num_steps)            

            logger.logkv("losses/dist_entropy", dist_entropy)
            logger.logkv("losses/value_loss", value_loss)
            logger.logkv("losses/action_loss", action_loss)

            logger.logkv("train/mean_episode_reward", np.mean(episode_rewards))
            logger.logkv("train/median_episode_reward", np.median(episode_rewards))

            # artificial_eval_episode_rewards = evaluate(args, eval_log_dir, actor_critic, device, env_type='artificial')
            # logger.logkv("test/artificial_mean_episode_reward", np.mean(artificial_eval_episode_rewards))
            # logger.logkv("test/artificial_median_episode_reward", np.median(artificial_eval_episode_rewards))

            # natural_eval_episode_rewards = evaluate(args, eval_log_dir, actor_critic, device, env_type='natural')
            # logger.logkv("test/natural_mean_episode_reward", np.mean(natural_eval_episode_rewards))
            # logger.logkv("test/natural_median_episode_reward", np.median(natural_eval_episode_rewards))

            logger.dumpkvs()


if __name__ == "__main__":
    main()
