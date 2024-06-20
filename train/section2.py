import torch
import numpy as np
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
import sys
import glob

try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
except IndexError:
    pass

import utils.carla_utils as cu
from models.image import ImageModelSS
from models.birdview import BirdEyeModelPolicy

CROP_SIZE = 192
PIXELS_PER_METER = 5

def repeat(a, repeats, dim=0):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = repeats
    a = a.repeat(*repeat_idx)
    order_index = torch.arange(init_dim, device=a.device).repeat(repeats) + (torch.arange(repeats, device=a.device) * init_dim).view(-1, 1)
    return a.index_select(dim, order_index.view(-1))

def get_weight(learner_points, teacher_points):
    decay = torch.FloatTensor([0.7**i for i in range(5)]).to(learner_points.device)
    xy_bias = torch.FloatTensor([0.7, 0.3]).to(learner_points.device)
    loss_weight = torch.mean((torch.abs(learner_points - teacher_points) * xy_bias).sum(dim=-1) * decay, dim=-1)
    x_weight = torch.max(torch.mean(teacher_points[..., 0], dim=-1), torch.mean(teacher_points[..., 0] * -1.4, dim=-1))
    return loss_weight

def weighted_random_choice(weights):
    return np.searchsorted(np.cumsum(weights), random.uniform(0, np.sum(weights)))

def get_optimizer(parameters, lr=1e-4):
    return torch.optim.Adam(parameters, lr=lr)

def load_image_model(backbone, ckpt, device='cuda'):
    net = ImageModelSS(backbone, all_branch=True).to(device)
    net.load_state_dict(torch.load(ckpt))
    return net

def _log_visuals(rgb_image, birdview, speed, command, loss, pred_locations, _pred_locations, _teac_locations, size=16):
    import cv2
    import utils.carla_utils as cu

    WHITE, BLUE, RED = [255, 255, 255], [0, 0, 255], [255, 0, 0]
    _numpy = lambda x: x.detach().cpu().numpy().copy()
    images = []

    for i in range(min(birdview.shape[0], size)):
        loss_i = loss[i].sum()
        canvas = cu.visualize_birdview(np.uint8(_numpy(birdview[i]).transpose(1, 2, 0) * 255).copy())
        rgb = np.uint8(_numpy(rgb_image[i]).transpose(1, 2, 0) * 255).copy()
        rows = [x * (canvas.shape[0] // 10) for x in range(11)]
        cols = [x * (canvas.shape[1] // 10) for x in range(11)]

        def _write(text, i, j):
            cv2.putText(canvas, text, (cols[j], rows[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        def _dot(canvas, i, j, color, radius=2):
            x, y = int(j), int(i)
            canvas[x-radius:x+radius+1, y-radius:y+radius+1] = color

        def _stick_together(a, b):
            h = min(a.shape[0], b.shape[0])
            r1, r2 = h / a.shape[0], h / b.shape[0]
            a = cv2.resize(a, (int(r1 * a.shape[1]), h))
            b = cv2.resize(b, (int(r2 * b.shape[1]), h))
            return np.concatenate([a, b], 1)

        _command = {1: 'LEFT', 2: 'RIGHT', 3: 'STRAIGHT', 4: 'FOLLOW'}.get(torch.argmax(command[i]).item() + 1, '???')
        _dot(canvas, 0, 0, WHITE)

        for x, y in (_teac_locations[i] + 1) * (0.5 * CROP_SIZE): _dot(canvas, x, y, BLUE)
        for x, y in _pred_locations[i]: _dot(rgb, x, y, RED)
        for x, y in pred_locations[i]: _dot(canvas, x, y, RED)

        _write(f'Command: {_command}', 1, 0)
        _write(f'Loss: {loss[i].item():.2f}', 2, 0)
        images.append((loss[i].item(), _stick_together(rgb, canvas)))

    return [x[1] for x in sorted(images, reverse=True, key=lambda x: x[0])]

def load_birdview_model(backbone, ckpt, device='cuda'):
    net = BirdEyeModelPolicy(backbone, all_branch=True).to(device)
    net.load_state_dict(torch.load(ckpt))
    return net

class CoordConverter():
    def __init__(self, w=384, h=160, fov=90, world_y=1.4, fixed_offset=4.0, device='cuda'):
        self._img_size = torch.FloatTensor([w, h]).to(device)
        self._fov = fov
        self._world_y = world_y
        self._fixed_offset = fixed_offset

    def __call__(self, camera_locations):
        camera_locations = (camera_locations + 1) * self._img_size / 2
        w, h = int(self._img_size[0]), int(self._img_size[1])
        cx, cy = w / 2, h / 2
        f = w / (2 * np.tan(self._fov * np.pi / 360))
        xt = (camera_locations[..., 0] - cx) / f
        yt = (camera_locations[..., 1] - cy) / f
        world_z = self._world_y / yt
        world_x = world_z * xt
        map_output = torch.stack([world_x, world_z], dim=-1) if isinstance(camera_locations, torch.Tensor) else np.stack([world_x, world_z], axis=-1)
        map_output *= PIXELS_PER_METER
        map_output[..., 1] = CROP_SIZE - map_output[..., 1]
        map_output[..., 0] += CROP_SIZE / 2
        map_output[..., 1] += self._fixed_offset * PIXELS_PER_METER
        return map_output

class LocationLoss(torch.nn.Module):
    def forward(self, pred_locations, teac_locations):
        pred_locations = pred_locations / (0.5 * CROP_SIZE) - 1
        return torch.mean(torch.abs(pred_locations - teac_locations), dim=(1, 2, 3))

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, buffer_limit=100000, augment=None, sampling=True, aug_fix_iter=1000000, batch_aug=4):
        self.buffer_limit = buffer_limit
        self._data = []
        self._weights = []
        self.rgb_transform = transforms.ToTensor()
        self.birdview_transform = transforms.ToTensor()
        self.augmenter = getattr(augmenter, augment) if augment and augment != 'None' else None
        self.normalized = False
        self._sampling = sampling
        self.aug_fix_iter = aug_fix_iter
        self.batch_aug = batch_aug

    def __len__(self):
        return len(self._data)

    def __getitem__(self, _idx):
        idx = weighted_random_choice(self._weights) if self._sampling and self.normalized else _idx
        rgb_img, cmd, speed, target, birdview_img = self._data[idx]
        rgb_imgs = [self.augmenter(self.aug_fix_iter).augment_image(rgb_img) if self.augmenter else rgb_img for _ in range(self.batch_aug)]
        rgb_imgs = [self.rgb_transform(img) for img in rgb_imgs]
        rgb_imgs = rgb_imgs[0] if self.batch_aug == 1 else torch.stack(rgb_imgs)
        birdview_img = self.birdview_transform(birdview_img)
        return idx, rgb_imgs, cmd, speed, target, birdview_img

    def update_weights(self, idxes, losses):
        for idx, loss in zip(idxes.numpy(), losses.detach().cpu().numpy()):
            if idx < len(self._data):
                self._new_weights[idx] = loss

    def init_new_weights(self):
        self._new_weights = self._weights.copy()

    def normalize_weights(self):
        self._weights = self._new_weights
        self.normalized = True

    def add_data(self, rgb_img, cmd, speed, target, birdview_img, weight):
        self.normalized = False
        self._data.append((rgb_img, cmd, speed, target, birdview_img))
        self._weights.append(weight)
        if len(self._data) > self.buffer_limit:
            idx = np.argmin(self._weights)
            self._data.pop(idx)
            self._weights.pop(idx)

    def remove_data(self, idx):
        self._weights.pop(idx)
        self._data.pop(idx)

    def get_highest_k(self, k):
        top_idxes = np.argsort(self._weights)[-k:]
        data = [self._data[idx] for idx in top_idxes if idx < len(self._data)]
        rgb_images = torch.stack([TF.to_tensor(np.ascontiguousarray(rgb_img)) for rgb_img, _, _, _, _ in data])
        bird_views = torch.stack([TF.to_tensor(birdview_img) for _, _, _, _, birdview_img in data])
        cmds = torch.FloatTensor([cmd for _, cmd, _, _, _ in data])
        speeds = torch.FloatTensor([speed for _, _, speed, _, _ in data])
        targets = torch.FloatTensor([target for _, _, _, target, _ in data])
        return rgb_images, bird_views, cmds, speeds, targets
