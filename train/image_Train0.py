import time
import argparse
from pathlib import Path

import numpy as np
import torch
import tqdm
import glob
import os
import sys
import cv2

try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
except IndexError:
    pass

import utils.bz_utils as bzu
from models.birdview import BirdEyeModelPolicy
from models.image import ImageModel
from train_util import one_hot
from utils.datasets.image_lmdb import get_image as load_data

BACKBONE = 'resnet34'
GAP = 5
N_STEP = 5
PIXELS_PER_METER = 5
CROP_SIZE = 192
SAVE_EPOCHS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1000]


class CoordConverter:
    def __init__(self, w=384, h=160, fov=90, world_y=1.4, fixed_offset=4.0, device='cuda'):
        self._w = w
        self._h = h
        self._img_size = torch.FloatTensor([w, h]).to(device)
        self._fov = fov
        self._world_y = world_y
        self._fixed_offset = fixed_offset

        self._tran = np.array([0.0, 0.0, 0.0])
        self._rot = np.array([0.0, 0.0, 0.0])
        f = self._w / (2 * np.tan(self._fov * np.pi / 360))
        self._A = np.array([
            [f, 0.0, self._w / 2],
            [0, f, self._h / 2],
            [0.0, 0.0, 1.0]
        ])

    def _project_image_xy(self, xy):
        N = len(xy)
        xyz = np.zeros((N, 3))
        xyz[:, 0] = xy[:, 0]
        xyz[:, 1] = self._world_y
        xyz[:, 2] = xy[:, 1]

        image_xy, _ = cv2.projectPoints(xyz, self._tran, self._rot, self._A, None)
        image_xy[..., 0] = np.clip(image_xy[..., 0], 0, self._w)
        image_xy[..., 1] = np.clip(image_xy[..., 1], 0, self._h)

        return image_xy[:, 0]

    def __call__(self, map_locations):
        teacher_locations = map_locations.detach().cpu().numpy()
        teacher_locations = (teacher_locations + 1) * CROP_SIZE / 2
        N = teacher_locations.shape[0]
        teacher_locations[:, :, 1] = CROP_SIZE - teacher_locations[:, :, 1]
        teacher_locations[:, :, 0] -= CROP_SIZE / 2
        teacher_locations = teacher_locations / PIXELS_PER_METER
        teacher_locations[:, :, 1] += self._fixed_offset
        teacher_locations = self._project_image_xy(np.reshape(teacher_locations, (N * N_STEP, 2)))
        teacher_locations = np.reshape(teacher_locations, (N, N_STEP, 2))
        teacher_locations = torch.FloatTensor(teacher_locations)

        return teacher_locations


class LocationLoss(torch.nn.Module):
    def __init__(self, w=384, h=160, device='cuda', **kwargs):
        super().__init__()
        self._img_size = torch.FloatTensor([w, h]).to(device)

    def forward(self, pred_locations, locations):
        locations = locations.to(pred_locations.device)
        locations = locations / (0.5 * self._img_size) - 1
        return torch.mean(torch.abs(pred_locations - locations), dim=(1, 2))


def log_visuals(rgb_image, birdview, speed, command, loss, pred_locations, teac_locations, _teac_locations, size=32):
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
        rgb = np.uint8(_numpy(rgb_image[i]).transpose(1, 2, 0) * 255).copy()
        rows = [x * (canvas.shape[0] // 10) for x in range(11)]
        cols = [x * (canvas.shape[1] // 10) for x in range(11)]

        def write_text(text, i, j):
            cv2.putText(canvas, text, (cols[j], rows[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        def draw_dot(_canvas, i, j, color, radius=2):
            x, y = int(j), int(i)
            _canvas[x - radius:x + radius + 1, y - radius:y + radius + 1] = color

        def stick_together(a, b):
            h = min(a.shape[0], b.shape[0])
            a = cv2.resize(a, (int(h / a.shape[0] * a.shape[1]), h))
            b = cv2.resize(b, (int(h / b.shape[0] * b.shape[1]), h))
            return np.concatenate([a, b], 1)

        command_text = {1: 'LEFT', 2: 'RIGHT', 3: 'STRAIGHT', 4: 'FOLLOW'}.get(torch.argmax(command[i]).item() + 1, '???')

        draw_dot(canvas, 0, 0, WHITE)

        for x, y in (_teac_locations[i] + 1) * (0.5 * CROP_SIZE):
            draw_dot(canvas, x, y, BLUE)
        for x, y in teac_locations[i]:
            draw_dot(rgb, x, y, BLUE)
        for x, y in pred_locations[i]:
            draw_dot(rgb, x, y, RED)

        write_text(f'Command: {command_text}', 1, 0)
        write_text(f'Loss: {loss[i].item():.2f}', 2, 0)

        images.append((loss[i].item(), stick_together(rgb, canvas)))

    return [x[1] for x in sorted(images, reverse=True, key=lambda x: x[0])]


def train_or_eval(coord_converter, criterion, net, teacher_net, data, optim, is_train, config, is_first_epoch):
    net.train() if is_train else net.eval()
    desc = 'Train' if is_train else 'Val'
    total = 10 if is_first_epoch else len(data)
    iterator_tqdm = tqdm.tqdm(data, desc=desc, total=total)
    iterator = enumerate(iterator_tqdm)
    tick = time.time()

    for i, (rgb_image, birdview, location, command, speed) in iterator:
        rgb_image, birdview, command, speed, location = map(lambda x: x.to(config['device']), (rgb_image, birdview, one_hot(command), speed, location.float()))

        with torch.no_grad():
            teac_location = teacher_net(birdview, speed, command)

        pred_location = net(rgb_image, speed, command)
        pred_location_img = (pred_location + 1) * coord_converter._img_size / 2
        teac_location_img = coord_converter(teac_location)

        loss = criterion(pred_location, teac_location_img)
        loss_mean = loss.mean()

        if is_train and not is_first_epoch:
            optim.zero_grad()
            loss_mean.backward()
            optim.step()

        if i % config['log_iterations'] == 0 or not is_train or is_first_epoch:
            images = log_visuals(rgb_image, birdview, speed, command, loss, pred_location_img, teac_location_img, teac_location)
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

    data_train, data_val = load_data(**config['data_args'])
    criterion = LocationLoss(**config['camera_args'])
    net = ImageModel(
        config['model_args']['backbone'],
        pretrained=config['model_args']['imagenet_pretrained']
    ).to(config['device'])
    teacher_net = BirdEyeModelPolicy(teacher_config['model_args']['backbone']).to(config['device'])
    teacher_net.load_state_dict(torch.load(config['teacher_args']['model_path']))
    teacher_net.eval()

    coord_converter = CoordConverter(**config['camera_args'])
    optim = torch.optim.Adam(net.parameters(), lr=config['optimizer_args']['lr'])

    for epoch in tqdm.tqdm(range(config['max_epoch'] + 1), desc='Epoch'):
        train_or_eval(coord_converter, criterion, net, teacher_net, data_train, optim, True, config, epoch == 0)
        train_or_eval(coord_converter, criterion, net, teacher_net, data_val, None, False, config, epoch == 0)

        if epoch in SAVE_EPOCHS:
            torch.save(net.state_dict(), str(Path(config['log_dir']) / f'model-{epoch}.th'))

        bzu.log.end_epoch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--log_iterations', type=int, default=1000)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--teacher_path', required=True)
    parser.add_argument('--fixed_offset', type=float, default=4.0)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--augment', choices=['None', 'medium', 'medium_harder', 'super_hard'], default=None)
    parser.add_argument('--lr', type=float, default=1e-4)

    parsed = parser.parse_args()

    config = {
        'log_dir': parsed.log_dir,
        'log_iterations': parsed.log_iterations,
        'max_epoch': parsed.max_epoch,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'optimizer_args': {'lr': parsed.lr},
        'data_args': {
            'batch_size': parsed.batch_size,
            'n_step': N_STEP,
            'gap': GAP,
            'augment': parsed.augment,
            'num_workers': 8,
        },
        'model_args': {
            'model': 'image_ss',
            'imagenet_pretrained': parsed.pretrained,
            'backbone': BACKBONE,
        },
        'camera_args': {
            'w': 384,
            'h': 160,
            'fov': 90,
            'world_y': 1.4,
            'fixed_offset': parsed.fixed_offset,
        },
        'teacher_args': {
            'model_path': parsed.teacher_path,
        }
    }

    train(config)
