import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from .resnet import get_resnet

CROP_SIZE = 192
MAP_SIZE = 320

def crop_bird_view(bird_view, dx=0, dy=0):
    x = 260 - CROP_SIZE // 2 + dx
    y = MAP_SIZE // 2 + dy
    return bird_view[x - CROP_SIZE // 2:x + CROP_SIZE // 2, y - CROP_SIZE // 2:y + CROP_SIZE // 2]

def choose_branch(branches, one_hot):
    for i, s in enumerate(branches.size()[2:]):
        one_hot = one_hot.unsqueeze(-1).expand_as(branches)
    return (one_hot * branches).sum(dim=1)

def calculate_signed_angle(u, v):
    theta = math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return theta if np.cross(u, v)[2] >= 0 else -theta

def project_to_circle(point, center, radius):
    direction = point - center
    return center + (direction / np.linalg.norm(direction)) * radius

def create_arc(points, center, radius):
    point_min = project_to_circle(points[0], center, radius)
    point_max = project_to_circle(points[-1], center, radius)
    theta_min, theta_max = np.arctan2(point_min[1], point_min[0]), np.arctan2(point_max[1], point_max[0])
    theta = np.linspace(theta_min, theta_max, 100)
    return np.stack([radius * np.cos(theta) + center[0], radius * np.sin(theta) + center[1]], axis=1)

class ResNetBase(nn.Module):
    def __init__(self, backbone, input_channel=3, bias_first=True, pretrained=False):
        super().__init__()
        self.conv, self.c = get_resnet(backbone, input_channel=input_channel, bias_first=bias_first, pretrained=pretrained)

class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(torch.FloatTensor(mean).view(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.FloatTensor(std).view(1, 3, 1, 1), requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std

class SpatialSoftmaxLayer(nn.Module):
    def __init__(self, height, width, channels, temperature=None):
        super().__init__()
        self.height, self.width, self.channels = height, width, channels
        self.temperature = nn.Parameter(torch.ones(1) * temperature) if temperature else 1.0
        pos_x, pos_y = np.meshgrid(np.linspace(-1., 1., height), np.linspace(-1., 1., width))
        self.register_buffer('pos_x', torch.FloatTensor(pos_x.ravel()))
        self.register_buffer('pos_y', torch.FloatFloatTensor(pos_y.ravel()))

    def forward(self, feature):
        feature = feature.view(-1, self.height * self.width)
        weight = F.softmax(feature / self.temperature, dim=-1)
        expected_x = (self.pos_x * weight).sum(dim=-1, keepdim=True)
        expected_y = (self.pos_y * weight).sum(dim=-1, keepdim=True)
        return torch.cat([expected_x, expected_y], dim=-1).view(-1, self.channels, 2)

class SpatialSoftmaxBZLayer(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height, self.width = height, width
        pos_x, pos_y = np.meshgrid(np.linspace(-1., 1., height), np.linspace(-1., 1., width))
        self.register_buffer('pos_x', torch.FloatTensor(pos_x.ravel()))
        self.register_buffer('pos_y', torch.FloatTensor(pos_y.ravel()))

    def forward(self, feature):
        feature = feature.view(feature.size(0), feature.size(1), -1)
        softmax = F.softmax(feature, dim=-1)
        expected_x = (self.pos_x * softmax).sum(dim=-1)
        expected_y = (self.pos_y * softmax).sum(dim=-1)
        return torch.stack([(-expected_y + 1) / 2.0, expected_x], dim=-1)
