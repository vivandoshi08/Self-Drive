import time
import argparse
from pathlib import Path
import numpy as np
import tqdm
import glob
import sys

try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
    sys.path.append(glob.glob('../')[0])
except IndexError:
    pass

from bird_view.carla_built_utils import carla_utils as cu
from train_util import one_hot
from benchmark import make_suite
import utils.bz_utils as bzu

BACKBONE = 'resnet34'
GAP = 5
N_STEP = 5
CROP_SIZE = 192
MAP_SIZE = 320
SAVE_EPISODES = list(range(20))

def crop_birdview(birdview, dx=0, dy=0):
    x = 260 - CROP_SIZE // 2 + dx
    y = MAP_SIZE // 2 + dy
    return birdview[x - CROP_SIZE // 2:x + CROP_SIZE // 2, y - CROP_SIZE // 2:y + CROP_SIZE // 2]

def get_control(agent_control, teacher_control, episode, beta=0.95):
    prob = 0.5 + 0.5 * (1 - beta ** episode)
    return agent_control if np.random.uniform(0, 1) < prob else teacher_control

def rollout(replay_buffer, coord_converter, net, teacher_net, episode, 
            image_agent_kwargs=dict(), birdview_agent_kwargs=dict(),
            episode_length=1000, n_vehicles=100, n_pedestrians=250, port=2000, planner="new"):

    from models.image import ImageAgent
    from models.birdview import BirdViewAgent
    
    decay = np.array([0.7 ** i for i in range(5)])
    xy_bias = np.array([0.7, 0.3])
    weathers = list(cu.TRAIN_WEATHERS.keys())

    def _get_weight(a, b):
        loss_weight = np.mean((np.abs(a - b) * xy_bias).sum(axis=-1) * decay, axis=-1)
        x_weight = np.maximum(np.mean(a[..., 0], axis=-1), np.mean(a[..., 0] * -1.4, axis=-1))
        return loss_weight

    num_data = 0
    progress = tqdm.tqdm(range(episode_length * len(weathers)), desc='Frame')

    for weather in weathers:
        data = list()
        while len(data) < episode_length:
            with make_suite('FullTown01-v1', port=port, planner=planner) as env:
                start, target = env.pose_tasks[np.random.randint(len(env.pose_tasks))]
                env.init(weather=weather, start=start, target=target, n_pedestrians=n_pedestrians, n_vehicles=n_vehicles)
                env.success_dist = 5.0

                image_agent_kwargs['model'] = net
                birdview_agent_kwargs['model'] = teacher_net

                image_agent = ImageAgent(**image_agent_kwargs)
                birdview_agent = BirdViewAgent(**birdview_agent_kwargs)

                while not env.is_success() and not env.collided:
                    env.tick()
                    observations = env.get_observations()
                    image_control, _image_points = image_agent.run_step(observations, teaching=True)
                    _image_points = coord_converter(_image_points)
                    image_points = _image_points / (0.5 * CROP_SIZE) - 1
                    birdview_control, birdview_points = birdview_agent.run_step(observations, teaching=True)
                    weight = _get_weight(birdview_points, image_points)

                    control = get_control(image_control, birdview_control, episode)
                    env.apply_control(control)

                    data.append({
                        'rgb_img': observations["rgb"].copy(),
                        'cmd': int(observations["command"]),
                        'speed': np.linalg.norm(observations["velocity"]),
                        'target': birdview_points,
                        'weight': weight,
                        'birdview_img': crop_birdview(observations['birdview'], dx=-10),
                    })
                    
                    progress.update(1)
                    if len(data) >= episode_length:
                        break

                if env.collided:
                    data = data[:-5]

        for datum in data:
            replay_buffer.add_data(**datum)
            num_data += 1

def train_or_eval(coord_converter, criterion, net, teacher_net, data, optim, is_train, config, is_first_epoch):
    if is_train:
        desc = 'Train'
        net.train()
    else:
        desc = 'Val'
        net.eval()

    total = 10 if is_first_epoch else len(data)
    iterator_tqdm = tqdm.tqdm(data, desc=desc, total=total)
    iterator = enumerate(iterator_tqdm)
    tick = time.time()
    
    import torch.distributions as tdist
    noiser = tdist.Normal(torch.tensor(0.0), torch.tensor(config['speed_noise']))

    for i, (rgb_image, birdview, location, command, speed) in iterator:
        rgb_image, birdview, command, speed = map(lambda x: x.to(config['device']), (rgb_image, birdview, one_hot(command), speed))

        if is_train and config['speed_noise'] > 0:
            speed += noiser.sample(speed.size()).to(speed.device)
            speed = torch.clamp(speed, 0, 10)

        if len(rgb_image.size()) > 4:
            B, batch_aug, c, h, w = rgb_image.size()
            rgb_image = rgb_image.view(B * batch_aug, c, h, w)
            birdview = repeat(birdview, batch_aug)
            command = repeat(command, batch_aug)
            speed = repeat(speed, batch_aug)

        with torch.no_grad():
            _teac_location, _teac_locations = teacher_net(birdview, speed, command)

        _pred_location, _pred_locations = net(rgb_image, speed, command)
        pred_location = coord_converter(_pred_location)
        pred_locations = coord_converter(_pred_locations)

        loss = criterion(pred_locations, _teac_locations)
        loss_mean = loss.mean()

        if is_train and not is_first_epoch:
            optim.zero_grad()
            loss_mean.backward()
            optim.step()

        if i % config['log_iterations'] == 0 or not is_train or is_first_epoch:
            images = log_visuals(rgb_image, birdview, speed, command, loss, pred_location, (_pred_location + 1) * coord_converter._img_size / 2, _teac_location)
            bzu.log.scalar(is_train=is_train, loss_mean=loss_mean.item())
            bzu.log.image(is_train=is_train, birdview=images)

        bzu.log.scalar(is_train=is_train, fps=1.0 / (time.time() - tick))
        tick = time.time()

        if is_first_epoch and i == 10:
            iterator_tqdm.close()
            break

def train(config):
    bzu.log.init(config['log_dir'])
    bzu.log.save_config(config)
    teacher_config = bzu.log.load_config(config['teacher_args']['model_path'])

    from phase2_utils import (
        CoordConverter, 
        ReplayBuffer, 
        LocationLoss, 
        load_birdview_model,
        load_image_model,
        get_optimizer
    )

    criterion = LocationLoss()
    net = load_image_model(config['model_args']['backbone'], config['phase1_ckpt'], device=config['device'])
    teacher_net = load_birdview_model(teacher_config['model_args']['backbone'], config['teacher_args']['model_path'], device=config['device'])

    image_agent_kwargs = {'camera_args': config["agent_args"]['camera_args']}
    coord_converter = CoordConverter(**config["agent_args"]['camera_args'])
    replay_buffer = ReplayBuffer(**config["buffer_args"])
        
    for episode in tqdm.tqdm(range(config['max_episode']), desc='Episode'):
        rollout(replay_buffer, coord_converter, net, teacher_net, episode, image_agent_kwargs=image_agent_kwargs, port=config['port'])
        _train(replay_buffer, net, teacher_net, criterion, coord_converter, bzu.log, config, episode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--log_iterations', default=100)
    parser.add_argument('--max_episode', default=20)
    parser.add_argument('--epoch_per_episode', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--speed_noise', type=float, default=0.0)
    parser.add_argument('--batch_aug', type=int, default=1)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--teacher_path', required=True)
    parser.add_argument('--fixed_offset', type=float, default=4.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--port', type=int, default=2000)

    parsed = parser.parse_args()

    config = {
        'port': parsed.port,
        'log_dir': parsed.log_dir,
        'log_iterations': parsed.log_iterations,
        'batch_size': parsed.batch_size,
        'max_episode': parsed.max_episode,
        'speed_noise': parsed.speed_noise,
        'epoch_per_episode': parsed.epoch_per_episode,
        'device': 'cuda',
        'phase1_ckpt': parsed.ckpt,
        'optimizer_args': {'lr': parsed.lr},
        'buffer_args': {
            'buffer_limit': 200000,
            'batch_aug': parsed.batch_aug,
            'augment': 'super_hard',
            'aug_fix_iter': 819200,
        },
        'model_args': {
            'model': 'image_ss',
            'backbone': BACKBONE,
        },
        'agent_args': {
            'camera_args': {
                'w': 384,
                'h': 160,
                'fov': 90,
                'world_y': 1.4,
                'fixed_offset': parsed.fixed_offset,
            }
        },
        'teacher_args': {
            'model_path': parsed.teacher_path,
        }
    }

    train(config)
