import numpy as np
import torch
import torchvision.transforms as T
import carla

class VehicleAgent:
    def __init__(self, neural_model, **extra_params):
        if neural_model is None:
            raise ValueError("Model cannot be None")

        if extra_params:
            print('Unused parameters: {}'.format(extra_params))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = T.ToTensor()
        self.identity_matrix = torch.FloatTensor(torch.eye(4))

        self.model = neural_model.to(self.device)
        self.model.eval()

        self.debug_info = {}

    def adjust_controls(self, steer_val, throttle_val, brake_val):
        vehicle_control = carla.VehicleControl()
        vehicle_control.steer = np.clip(steer_val, -1.0, 1.0)
        vehicle_control.throttle = np.clip(throttle_val, 0.0, 1.0)
        vehicle_control.brake = np.clip(brake_val, 0.0, 1.0)
        vehicle_control.manual_gear_shift = False

        return vehicle_control
