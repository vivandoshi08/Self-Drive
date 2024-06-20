import cv2
import numpy as np
import torch
import torch.nn as nn
from . import common
from .agent import Agent
from .controller import PIDController, CustomController, ls_circle

STEPS = 5
SPEED_STEPS = 3
COMMANDS = 4
DT = 0.1
CROP_SIZE = 192
PIXELS_PER_METER = 5

def create_regression_base():
    return nn.Sequential(
        nn.ConvTranspose2d(640, 256, 4, 2, 1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )

def create_spatial_softmax_base():
    return nn.Sequential(
        nn.BatchNorm2d(640),
        nn.ConvTranspose2d(640, 256, 3, 2, 1, 1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
        nn.ReLU(inplace=True)
    )

class BirdEyeModelPolicy(common.ResnetBase):
    def __init__(self, backbone='resnet18', input_channel=7, n_step=5, all_branch=False):
        super().__init__(backbone=backbone, input_channel=input_channel, bias_first=False)
        self.deconv = create_spatial_softmax_base()
        self.location_pred = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64, STEPS, 1),
                common.SpatialSoftmax(48, 48, STEPS)
            ) for _ in range(COMMANDS)
        ])
        self.all_branch = all_branch

    def forward(self, bird_view, velocity, command):
        h = self.conv(bird_view)
        velocity = velocity[..., None, None, None].repeat(1, 128, *h.shape[2:])
        h = torch.cat((h, velocity), dim=1)
        h = self.deconv(h)

        location_preds = torch.stack([pred(h) for pred in self.location_pred], dim=1)
        location_pred = common.select_branch(location_preds, command)
        return (location_pred, location_preds) if self.all_branch else location_pred

class BirdviewAgent(Agent):
    def __init__(self, steer_points=None, pid=None, gap=5, **kwargs):
        super().__init__(**kwargs)
        self.speed_control = PIDController(K_P=1.0, K_I=0.1, K_D=2.5)
        self.steer_points = steer_points or {"1": 3, "2": 2, "3": 2, "4": 2}
        self.turn_control = CustomController(pid or {
            "1": {"Kp": 1.0, "Ki": 0.1, "Kd": 0},
            "2": {"Kp": 1.0, "Ki": 0.1, "Kd": 0},
            "3": {"Kp": 0.8, "Ki": 0.1, "Kd": 0},
            "4": {"Kp": 0.8, "Ki": 0.1, "Kd": 0}
        })
        self.gap = gap

    def run_step(self, observations, teaching=False):
        birdview = common.crop_birdview(observations['birdview'], dx=-10)
        speed = np.linalg.norm(observations['velocity'])
        command = self.one_hot[int(observations['command']) - 1]

        with torch.no_grad():
            _birdview = self.transform(birdview).unsqueeze(0).to(self.device)
            _speed = torch.FloatTensor([speed]).to(self.device)
            _command = command.to(self.device).unsqueeze(0)
            _locations = self.model(_birdview, _speed, _command)[0].squeeze().cpu().numpy()

        _locations = (_locations + 1) / 2 * CROP_SIZE
        targets = [
            [np.linalg.norm([dx - CROP_SIZE / 2, CROP_SIZE - dy]) / PIXELS_PER_METER * np.cos(np.arctan2(dx - CROP_SIZE / 2, CROP_SIZE - dy)),
             np.linalg.norm([dx - CROP_SIZE / 2, CROP_SIZE - dy]) / PIXELS_PER_METER * np.sin(np.arctan2(dx - CROP_SIZE / 2, CROP_SIZE - dy))]
            for dx, dy in _locations
        ]

        target_speed = sum(
            np.linalg.norm([_locations[i][0] - _locations[i-1][0], _locations[i][1] - _locations[i-1][1]]) / 
            (PIXELS_PER_METER * self.gap * DT) / (SPEED_STEPS - 1) 
            for i in range(1, SPEED_STEPS)
        )

        _cmd = int(observations['command'])
        n = self.steer_points.get(str(_cmd), 1)
        targets = np.vstack([[[0, 0]], targets])
        closest = common.project_point_to_circle(targets[n], *ls_circle(targets))

        steer = self.turn_control.run_step(common.signed_angle([1.0, 0.0, 0.0], [closest[0], closest[1], 0.0]), _cmd)
        throttle = self.speed_control.step(target_speed - speed)
        brake = 1.0 if target_speed < 1.0 else 0.0

        self.debug.update({
            'locations_birdview': _locations[:, ::-1].astype(int),
            'target': closest,
            'target_speed': target_speed
        })

        control = self.postprocess(steer, throttle, brake)
        return (control, _locations) if teaching else control
