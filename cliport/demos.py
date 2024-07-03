"""Data collection script."""

import os
import hydra
import numpy as np
import random

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment

from PIL import Image
import numpy as np


@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    data_path = os.path.join(
        cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'val':  # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    # Collect training data from oracle demonstrations.
    if cfg.expert:
        base_pth: str = './expert_trajs'
    else:
        base_pth: str = './random_trajs'

    print(f"Base path: {base_pth}")
    while dataset.n_episodes < cfg['n']:
        traj_dir: str = f"traj_{dataset.n_episodes}"
        os.makedirs(base_pth + '/' + traj_dir)
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print(
            'Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))

        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'val' and seed > (-1 + 10000):
            raise Exception(
                "!!! Seeds for val set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f'{dataset.n_episodes+1:06d}')

        # Rollout expert policy
        for _ in range(task.max_steps):
            frame = env.render()
            Image.fromarray(frame).save(
                base_pth + '/' + traj_dir + '/' + f'img_{_}')
            act = agent.act(obs, info)
            if not cfg.dataset.expert:
                print(f"get here")
                act_pos0 = act['pose0']
                act_pos1 = act['pose1']
                # act_pick = act['pick']
                # act_place = act['place']
                act = {'pose0': (np.clip(act_pos0[0] + np.random.uniform(-0.25, 0.25, 3), -1, 1),
                                 np.clip(act_pos0[1] + np.random.uniform(-0.25, 0.25, 4), -1, 1)),
                       'pose1': (np.clip(act_pos1[0] + np.random.uniform(-0.25, 0.25, 3), -1, 1),
                                 np.clip(act_pos1[1] + np.random.uniform(-0.25, 0.25, 4), -1, 1))}
                       # 'pick': [np.clip(act_pick[0] + np.random.randint(0, 2), 0, 255),
                       #          np.clip(act_pick[1] + np.random.randint(0, 2), 0, 255), 0.0],
                       # 'place': [np.clip(act_place[0] + np.random.randint(0, 2), 0, 255),
                       #           np.clip(act_place[1] + np.random.randint(0, 2), 0, 255), 0.0]}

            episode.append((obs, act, reward, info))
            lang_goal = info['lang_goal']
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(
                f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break
        episode.append((obs, None, reward, info))
        dataset.n_episodes += 1

        # End video recording
        if record:
            env.end_rec()

        # Only save completed demonstrations.
        if save_data and total_reward > 0.99:
            dataset.add(seed, episode)


if __name__ == '__main__':
    main()
