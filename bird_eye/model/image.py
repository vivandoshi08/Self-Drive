import math
import numpy as np
import torch
import torch.nn as nn
from . import common
from .agent import Agent
from .controller import CustomController, PIDController, ls_circle

CROP_SIZE = 192
STEPS = 5
COMMANDS = 4
DT = 0.1
PIXELS_PER_METER = 5

class ImageModel(common.ResNetBase):
    def __init__(self, backbone, warp=False, pretrained=False, all_branch=False):
        super().__init__(backbone, pretrained=pretrained, input_channel=3, bias_first=False)
        self.channel_sizes = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048
        }
        self.warp = warp
        self.all_branch = all_branch
        self.rgb_transform = common.NormalizeV2(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.deconv = self._create_deconv_layers(self.channel_sizes[backbone])
        self.location_pred = self._create_location_pred_layers()

    def _create_deconv_layers(self, channels):
        return nn.Sequential(
            nn.BatchNorm2d(channels + 128),
            nn.ConvTranspose2d(channels + 128, 256, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
        )

    def _create_location_pred_layers(self):
        ow, oh = (48, 48) if self.warp else (96, 40)
        return nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64, STEPS, 1, 1, 0),
                common.SpatialSoftmax(ow, oh, STEPS),
            ) for _ in range(COMMANDS)
        ])

    def forward(self, image, velocity, command):
        if self.warp:
            warped_image = tgm.warp_perspective(image, self.M, dsize=(192, 192))
            resized_image = resize_images(image)
            image = torch.cat([warped_image, resized_image], 1)
        
        image = self.rgb_transform(image)
        h = self.conv(image)
        velocity = velocity[..., None, None, None].repeat((1, 128, h.size(2), h.size(3)))
        h = torch.cat((h, velocity), dim=1)
        h = self.deconv(h)
        
        location_preds = torch.stack([pred(h) for pred in self.location_pred], dim=1)
        location_pred = common.choose_branch(location_preds, command)
        return (location_pred, location_preds) if self.all_branch else location_pred

class ImageAgent(Agent):
    def __init__(self, steer_points=None, pid=None, gap=5, camera_args=None, **kwargs):
        super().__init__(**kwargs)
        self.fixed_offset = float(camera_args.get('fixed_offset', 4.0))
        self.img_size = np.array([float(camera_args.get('x', 384)), float(camera_args.get('h', 160))])
        self.gap = gap
        
        self.steer_points = steer_points or {"1": 4, "2": 3, "3": 2, "4": 2}
        self.turn_control = CustomController(pid or {
            "1": {"Kp": 0.5, "Ki": 0.20, "Kd": 0.0},
            "2": {"Kp": 0.7, "Ki": 0.10, "Kd": 0.0},
            "3": {"Kp": 1.0, "Ki": 0.10, "Kd": 0.0},
            "4": {"Kp": 1.0, "Ki": 0.50, "Kd": 0.0},
        })
        self.speed_control = PIDController(K_P=0.8, K_I=0.08, K_D=0.0)
        self.engine_brake_threshold = 2.0
        self.brake_threshold = 2.0
        self.last_brake = -1

    def run_step(self, observations, teaching=False):
        rgb = observations['rgb'].copy()
        speed = np.linalg.norm(observations['velocity'])
        command = self.one_hot[int(observations['command']) - 1]

        with torch.no_grad():
            _rgb = self.transform(rgb).to(self.device).unsqueeze(0)
            _speed = torch.FloatTensor([speed]).to(self.device)
            _command = command.to(self.device).unsqueeze(0)
            model_pred = self.model(_rgb, _speed, _command)[0] if self.model.all_branch else self.model(_rgb, _speed, _command)

        pixel_pred = model_pred.squeeze().detach().cpu().numpy()
        model_pred = (pixel_pred + 1) * self.img_size / 2
        world_pred = self.unproject(model_pred)

        targets = [(0, 0)]
        for i in range(STEPS):
            dx, dy = world_pred[i]
            angle = np.arctan2(dx, dy)
            dist = np.linalg.norm([dx, dy])
            targets.append([dist * np.cos(angle), dist * np.sin(angle)])

        targets = np.array(targets)
        target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / (self.gap * DT)
        c, r = ls_circle(targets)
        closest = common.project_to_circle(targets[self.steer_points.get(str(int(observations['command'])), 1)], c, r)
        
        acceleration = np.clip(target_speed - speed, 0.0, 1.0)
        alpha = common.calculate_signed_angle([1.0, 0.0, 0.0], [closest[0], closest[1], 0.0])
        steer = self.turn_control.run_step(alpha, int(observations['command']))
        throttle = self.speed_control.step(acceleration)
        brake = 1.0 if target_speed <= self.brake_threshold else 0.0

        if target_speed <= self.engine_brake_threshold:
            steer = throttle = 0.0

        self.debug = {
            'target_speed': target_speed,
            'target': closest,
            'locations_world': targets,
            'locations_pixel': model_pred.astype(int),
        }
        
        control = self.postprocess(steer, throttle, brake)
        return (control, pixel_pred) if teaching else control

    def unproject(self, output, world_y=1.4, fov=90):
        cx, cy = self.img_size / 2
        f = self.img_size[0] / (2 * np.tan(fov * np.pi / 360))
        xt = (output[..., 0:1] - cx) / f
        yt = (output[..., 1:2] - cy) / f
        world_z = world_y / yt
        world_x = world_z * xt
        world_output = np.stack([world_x, world_z], axis=-1)
        if self.fixed_offset:
            world_output[..., 1] -= self.fixed_offset
        return world_output.squeeze()
