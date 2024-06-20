import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
import carla
from .resnet import get_resnet
from .common import select_branch, Normalize
from .agent import Agent

def create_branch(dropout_rate):
    return nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(256, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(256, 3)
    )

class ModifiedBaseline(nn.Module):
    def __init__(self, backbone='resnet18', dropout_rate=0.5):
        super().__init__()
        self.conv_layers, channels = get_resnet(backbone, input_channel=3)
        self.avg_pool = nn.AvgPool2d((40, 96))
        self.rgb_transform = Normalize(mean=[0.31, 0.33, 0.36], std=[0.18, 0.18, 0.19])

        self.speed_transform = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.fusion = nn.Sequential(
            nn.Linear(channels + 128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.speed_prediction = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )

        self.control_branches = nn.ModuleList([create_branch(dropout_rate) for _ in range(4)])

    def forward(self, images, speeds, commands):
        features = self.conv_layers(self.rgb_transform(images))
        features = self.avg_pool(features).view(features.size(0), -1)
        speed_features = self.speed_transform(speeds.unsqueeze(1))

        combined_features = torch.cat([features, speed_features], dim=1)
        joint_features = self.fusion(combined_features)

        branch_outputs = torch.stack([branch(joint_features) for branch in self.control_branches], dim=1)
        control_output = select_branch(branch_outputs, commands)
        speed_output = self.speed_prediction(joint_features)

        return control_output, speed_output

class EnhancedAgent(Agent):
    def run_step(self, observations):
        rgb_image = observations['rgb']
        current_speed = np.linalg.norm(observations['velocity'])
        command_type = self.one_hot[int(observations['command']) - 1]

        with torch.no_grad():
            tensor_rgb = self.transform(rgb_image).unsqueeze(0).to(self.device)
            tensor_speed = torch.tensor([current_speed], device=self.device)
            tensor_command = command_type.to(self.device)

            control_tensor, _ = self.model(tensor_rgb, tensor_speed, tensor_command)
            steer, throttle, brake = control_tensor.cpu().numpy().squeeze()

        if not hasattr(self, 'initial_steps'):
            self.initial_steps = 0

        if self.initial_steps < 20:
            throttle = 0.5
            brake = 0
            self.initial_steps += 1

        return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
