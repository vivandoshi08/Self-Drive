from collections import deque
import numpy as np
from scipy.special import comb
from scipy import interpolate

def fit_circle_least_squares(points):
    '''
    Input: Nx2 points
    Output: center (cx, cy) and radius r
    '''
    xs, ys = points[:, 0], points[:, 1]
    us, vs = xs - np.mean(xs), ys - np.mean(ys)

    Suu, Suv, Svv = np.sum(us**2), np.sum(us * vs), np.sum(vs**2)
    Suuu, Suvv, Svvv, Svuu = np.sum(us**3), np.sum(us * vs**2), np.sum(vs**3), np.sum(vs * us**2)

    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = np.array([0.5 * (Suuu + Suvv), 0.5 (Svvv + Svuu)])

    cx, cy = np.linalg.solve(A, b)
    r = np.sqrt(cx**2 + cy**2 + (Suu + Svv) / len(xs))

    return np.array([cx + np.mean(xs), cy + np.mean(ys)]), r

class PIDController:
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, fps=10, n=30):
        self.K_P, self.K_I, self.K_D = K_P, K_I, K_D
        self.dt = 1.0 / fps
        self.error_window = deque(maxlen=n)

    def step(self, error):
        self.error_window.append(error)
        integral = sum(self.error_window) * self.dt if len(self.error_window) > 1 else 0.0
        derivative = (self.error_window[-1] - self.error_window[-2]) / self.dt if len(self.error_window) > 1 else 0.0

        return self.K_P * error + self.K_I * integral + self.K_D * derivative

class AdaptiveController:
    def __init__(self, controller_params, k=0.5, n=2, wheelbase=2.89, dt=0.1):
        self.k, self.wheelbase, self.dt = k, wheelbase, dt
        self.controller_params = controller_params
        self.error_buffer = deque(maxlen=10)

    def run_step(self, alpha, cmd):
        self.error_buffer.append(alpha)
        integral = sum(self.error_buffer) * self.dt if len(self.error_buffer) > 1 else 0.0
        derivative = (self.error_buffer[-1] - self.error_buffer[-2]) / self.dt if len(self.error_buffer) > 1 else 0.0

        params = self.controller_params[str(cmd)]
        return params["Kp"] * alpha + params["Ki"] * integral + params["Kd"] * derivative
