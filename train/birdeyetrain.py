import time
import argparse
from pathlib import Path
import numpy as np
import torch
import tqdm
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
except IndexError:
    pass

import utils.bz_utils as bzu
from models.birdview import BirdEyeModelPolicy
from train_util import one_hot
from utils.datasets.birdview_lmdb import get_birdview as load_data

BACKBONE = 'resnet18'
GAP = 5
N_STEP = 5
SAVE_EPOCHS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1000]

class LocationLoss(torch.nn.Module):
    def __init__(self, w=192, h=192, choice='l2'):
        super(LocationLoss, self).__init__()
        self.loss = torch.nn.MSELoss() if choice == 'l2' else lambda a, b: torch.mean(torch.abs(a - b), dim=(1, 2))
        self.img_size = torch.FloatTensor([w, h]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, pred_location, gt_location):
        gt_location = gt_location / (0.5 * self.img_size) - 1.0
        return self.loss(pred_location, gt_location)

def log_visuals(birdview, speed, command, loss, locations, _locations, size=16):
    import cv2
    import numpy as np
    import utils.carla_utils as cu

    WHITE = [255, 255, 255]
    BLUE = [0, 0, 255]
    RED = [255, 0, 0]
    _numpy = lambda x: x.detach().cpu().numpy().copy()

    images = []

    for i in range(min(birdview.shape[0], size)):
        loss_i = loss[i].sum()
        canvas = np.uint8(_numpy(birdview[i]).transpose(1, 2, 0) * 255).copy()
        canvas = cu.visualize_birdview(canvas)
        rows = [x * (canvas.shape[0] // 10) for x in range(11)]
        cols = [x * (canvas.shape[1] // 10) for x in range(11)]

        def write_text(text, i, j):
            cv2.putText(canvas, text, (cols[j], rows[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        def draw_dot(i, j, color, radius=2):
            x, y = int(j), int(i)
            canvas[x-radius:x+radius+1, y-radius:y+radius+1] = color

        command_text = {1: 'LEFT', 2: 'RIGHT', 3: 'STRAIGHT', 4: 'FOLLOW'}.get(torch.argmax(command[i]).item() + 1, '???')
        draw_dot(0, 0, WHITE)

        for x, y in locations[i]:
            draw_dot(x, y, BLUE)
        for x, y in (_locations[i] + 1) * (0.5 * 192):
            draw_dot(x, y, RED)

        write_text(f'Command: {command_text}', 1, 0)
        write_text(f'Loss: {loss[i].item():.2f}', 2, 0)

        images.append((loss[i].item(), canvas))

    return [x[1] for x in sorted(images, reverse=True, key=lambda x: x[0])]

def train_or_eval(criterion, net, data, optim, is_train, config, is_first_epoch):
    net.train() if is_train else net.eval()
    desc = 'Train' if is_train else 'Val'
    total = 10 if is_first_epoch else len(data)
    iterator_tqdm = tqdm.tqdm(data, desc=desc, total=total)
    iterator = enumerate(iterator_tqdm)
    tick = time.time()

    for i, (birdview, location, command, speed) in iterator:
        birdview, command, speed, location = birdview.to(config['device']), one_hot(command).to(config['device']), speed.to(config['device']), location.float().to(config['device'])
        pred_location = net(birdview, speed, command)
        loss = criterion(pred_location, location)
        loss_mean = loss.mean()

        if is_train and not is_first_epoch:
            optim.zero_grad()
            loss_mean.backward()
            optim.step()

        should_log = i % config['log_iterations'] == 0 or not is_train or is_first_epoch

        if should_log:
            metrics = {'loss': loss_mean.item()}
            images = log_visuals(birdview, speed, command, loss, location, pred_location)
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

    data_train, data_val = load_data(**config['data_args'])
    criterion = LocationLoss(w=192, h=192, choice='l1')
    net = BirdEyeModelPolicy(config['model_args']['backbone']).to(config['device'])

    if config['resume']:
        checkpoint = sorted(Path(config['log_dir']).glob('model-*.th'))[-1]
        print(f"Loading {checkpoint}")
        net.load_state_dict(torch.load(checkpoint))

    optim = torch.optim.Adam(net.parameters(), lr=config['optimizer_args']['lr'])

    for epoch in tqdm.tqdm(range(config['max_epoch'] + 1), desc='Epoch'):
        train_or_eval(criterion, net, data_train, optim, True, config, epoch == 0)
        train_or_eval(criterion, net, data_val, None, False, config, epoch == 0)

        if epoch in SAVE_EPOCHS:
            torch.save(net.state_dict(), str(Path(config['log_dir']) / f'model-{epoch}.th'))

        bzu.log.end_epoch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--log_iterations', default=1000, type=int)
    parser.add_argument('--max_epoch', default=1000, type=int)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--x_jitter', type=int, default=5)
    parser.add_argument('--y_jitter', type=int, default=0)
    parser.add_argument('--angle_jitter', type=int, default=5)
    parser.add_argument('--gap', type=int, default=5)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--cmd_biased', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)

    parsed = parser.parse_args()
    config = {
        'log_dir': parsed.log_dir,
        'resume': parsed.resume,
        'log_iterations': parsed.log_iterations,
        'max_epoch': parsed.max_epoch,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'optimizer_args': {'lr': parsed.lr},
        'data_args': {
            'batch_size': parsed.batch_size,
            'n_step': N_STEP,
            'gap': GAP,
            'crop_x_jitter': parsed.x_jitter,
            'crop_y_jitter': parsed.y_jitter,
            'angle_jitter': parsed.angle_jitter,
            'max_frames': parsed.max_frames,
            'cmd_biased': parsed.cmd_biased,
        },
        'model_args': {
            'model': 'birdview_dian',
            'input_channel': 7,
            'backbone': BACKBONE,
        },
    }
    train(config)
