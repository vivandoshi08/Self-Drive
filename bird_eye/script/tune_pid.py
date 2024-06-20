import numpy as np
import tqdm
import carla
import matplotlib.pyplot as plt

from agents.navigation.roaming_agent import roaming
from bird_view.carla_built_utils import carla_utils as cu
from bird_view.carla_built_utils import bz_utils as bzu

TOWN = 'Town01'
PORT = 3000

TRAIN = [(25, 29), (28, 24), (99, 103), (144, 148), (151, 147)]
VAL = [(57, 39), (50, 48), (36, 53), (136, 79), (22, 76)]

def initialize_environment(town, port, spawn_point=15, weather='clear_noon', n_vehicles=0):
    params = {
        'spawn': spawn_point,
        'weather': weather,
        'n_vehicles': n_vehicles
    }
    return cu.CarlaWrapper(town, cu.VEHICLE_NAME, port), params

def world_loop(opts_dict):
    with initialize_environment(TOWN, PORT) as (env, params):
        env.init(**params)
        agent = roaming(env._player, False, opts_dict)

        for _ in range(30):
            env.tick()
            env.apply_control(agent.run_step()[0])

        for _ in tqdm.tqdm(range(125), desc='Simulation'):
            env.tick()
            observations = env.get_observations()
            inputs = cu.get_inputs(observations)
            debug = dict()
            control, command = agent.run_step(inputs, debug_info=debug)
            env.apply_control(control)

            observations.update({'control': control, 'command': command})
            processed = cu.process(observations)

            yield debug

            bzu.show_image('rgb', processed['rgb'])
            bzu.show_image('birdview', cu.visualize_birdview(processed['birdview']))

def main():
    plt.ion()
    np.random.seed(0)

    for _ in tqdm.tqdm(range(10000), desc='Trials'):
        desired, current, output, e = [], [], [], []

        K_P = np.random.uniform(0.5, 2.0)
        K_I = np.random.uniform(0.0, 2.0)
        K_D = np.random.uniform(0.0, 0.05)

        opts_dict = {
            'lateral_control_dict': {
                'K_P': K_P,
                'K_I': K_I,
                'K_D': K_D,
                'dt': 0.1
            }
        }

        for debug in world_loop(opts_dict):
            for lst in [desired, current, output]:
                if len(lst) > 500:
                    lst.pop(0)

            desired.append(debug['desired'])
            current.append(debug['current'])
            output.append(debug['output'])
            e.append(debug['e'] ** 2)

        plot_name = f'{sum(e):.1f}_{K_P:.3f}_{K_I:.3f}_{K_D:.3f}'

        plt.cla()
        plt.plot(range(len(desired)), desired, 'b-', label='Desired')
        plt.plot(range(len(current)), current, 'r-', label='Current')
        plt.plot(range(len(output)), output, 'c-', label='Output')
        plt.legend()
        plt.savefig(f'/home/vivandoshi/hd_data/images/{plot_name}.png')

if __name__ == '__main__':
    main()
