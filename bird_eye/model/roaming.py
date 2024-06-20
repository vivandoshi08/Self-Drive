import numpy as np
import carla
from agents.navigation.agent import Agent
from agents.navigation.local_planner import LocalPlannerNew
from .controller import PIDController

TURNING_PID = {
    'K_P': 1.5,
    'K_I': 0.5,
    'K_D': 0.0,
    'fps': 10
}

class RoamingAgent(Agent):
    def __init__(self, vehicle, resolution, threshold_before, threshold_after):
        super().__init__(vehicle)

        self.proximity_threshold = 9.5
        self.speed_control = PIDController(K_P=1.0)
        self.turn_control = PIDController(**TURNING_PID)
        self.local_planner = LocalPlannerNew(self._vehicle, resolution, threshold_before, threshold_after)
        self.set_route = self.local_planner.set_route
        self.debug = {}

    def run_step(self, inputs=None, debug=False, debug_info=None):
        self.local_planner.run_step()

        transform = self._vehicle.get_transform()
        forward_vector = transform.get_forward_vector()
        rot_matrix = np.array([[forward_vector.x, forward_vector.y], [-forward_vector.y, forward_vector.x]])

        target_location = self.local_planner.target[0].transform.location
        target_coords = np.array([target_location.x, target_location.y])
        vehicle_coords = np.array([self._vehicle.get_location().x, self._vehicle.get_location().y])
        diff_coords = rot_matrix.dot(target_coords - vehicle_coords)

        speed = np.linalg.norm([self._vehicle.get_velocity().x, self._vehicle.get_velocity().y])

        u = np.array([diff_coords[0], diff_coords[1], 0.0])
        v = np.array([1.0, 0.0, 0.0])
        theta = np.arccos(np.dot(u, v) / np.linalg.norm(u))
        theta = theta if np.cross(u, v)[2] < 0 else -theta
        steer = self.turn_control.step(theta)

        target_speed = 6.0
        if int(self.local_planner.target[1]) not in [3, 4]:
            target_speed *= 0.75

        throttle = self.speed_control.step(target_speed - speed)

        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = 0.0
        control.manual_gear_shift = False

        hazard_detected, hazard_actor = self.check_for_hazards()
        if hazard_detected:
            control = self.emergency_stop()
            control.manual_gear_shift = False
            return control

        self.debug['target'] = (hazard_actor.get_location().x, hazard_actor.get_location().y) if hazard_actor else (target_coords[0], target_coords[1])
        return control

    def check_for_hazards(self):
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter('*vehicle*')
        lights_list = actor_list.filter('*traffic_light*')
        walkers_list = actor_list.filter('*walker*')

        blocking_vehicle, vehicle = self._is_vehicle_hazard(vehicle_list)
        blocking_light, traffic_light = self._is_light_red(lights_list)
        blocking_walker, walker = self._is_walker_hazard(walkers_list)

        if blocking_vehicle:
            return True, vehicle
        if blocking_light:
            return True, traffic_light
        if blocking_walker:
            return True, walker

        return False, None
