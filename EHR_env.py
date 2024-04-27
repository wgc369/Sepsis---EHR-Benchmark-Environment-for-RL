import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils

import helpers

class EHREnv(gym.Env):
    def __init__(self, model='lstm', sofa=5):
        self.min_action = np.array([0.0, 0.0, 0.0, 0.0])#, dtype=np.float64
        self.max_action = np.array([100.0, 200.0, 4.0, 4.0])#, dtype=np.float64

        _temp_low_state = ['0.0000000000', '6582.4222337963', '0.0000000000', '0.0000000000', '0.0000000000', '3.0000000000', '0.0000000000', '0.0000000000', '0.0000000000', '-58.0000000000', '0.0000000000', '50.0000000000', '-17.7777777778', '0.2000000000', '1.5000000000', '96.0000000000', '70.0000000000', '16.0000000000', '0.0000000000', '0.0000000000', '0.0000000000', '0.0000000000', '-27.6333333333', '0.0000000000', '0.0000000000', '0.0000000000', '0.0000000000', '1.0000000000', '-0.4248777079', '0.0000000000', '5.0000000000', '0.0000000000', '7.5500000000', '0.0000000000', '6.7700000000', '16.0000000000', '0.0000000000', '-41.0000000000', '0.0000000000', '0.0000000000', '0.0000000000', '0.0000000000', '18.0000000000', '-327907.2833333330', '0.0000000000', '0.0000000000'] 
        _temp_low_state = [-float('inf') for i in range(46)]#[float(i) for i in _temp_low_state]
        self.low_state = np.array([i for i in _temp_low_state])#, dtype=np.float64

        _temp_high_state = ['1.0000000000', '33383.8500000000', '14.0000000000', '1.0000000000', '278.5000000000', '15.0000000000', '190.0000000000', '265.0000000000', '189.8750000000', '214.0714285714', '65.0000000000', '100.0000000000', '536.1111111111', '1.0000000000', '9.2454545455', '177.0000000000', '148.0000000000', '877.0000000000', '280.0000000000', '139.0000000000', '7.8000000000', '19.0000000000', '3.6300000000', '112.0000000000', '9939.0000000000', '9745.0000000000', '75.6000000000', '5.9000000000', '20.0000000000', '462.6000000000', '1980.0000000000', '162.0000000000', '193.0000000000', '18.0000000000', '7.7600000000', '607.0000000000', '179.2222222222', '77.8000000000', '29.3000000000', '172.0000000000', '1.0000000000', '1.6827579890', '2890.4761904762', '848814.9240000009', '23.0000000000', '4.0000000000']
        _temp_high_state = [float('inf') for i in range(46)]#[float(i) for i in _temp_high_state]
        self.high_state = np.array([i for i in _temp_high_state])#, dtype=np.float64

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(4,))#, dtype=np.float64
        
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, shape=(46,))#, dtype=np.float64


        self.EHRG = helpers.EHRGenerator(model, sofa)

    def step(self, action):
        prev_sofa = self.state[44]
        self.state = self.EHRG.get_next_state(self.state, action)

        reward = 0.0
        if self.state[44] > prev_sofa:
            reward = -1.0
        elif self.state[44] < prev_sofa:
            reward = 1.0

        terminated = False
        if self.state[44] == 0 or self.state[44] == 24:
            terminated = True
        # self.state = self.state.reshape(-1, 1)
        return self.state, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.state = self.EHRG.get_initial_state()
        # self.state = self.state.reshape(-1, 1)
        return np.array(self.state, dtype=np.float32), {} #, dtype=np.float64

    def close(self):
        pass