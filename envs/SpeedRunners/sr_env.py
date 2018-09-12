import numpy as np

from HLML.envs import Env
from HLML.envs.utils import ScreenViewer
from HLML.envs.utils import MemoryReader

from .actor import Actor

class SpeedRunnersEnv(Env):
    def __init__(self, max_time, image_shape):
        Env.__init__(self, max_time)

        self.actor = Actor()
        self.sv = ScreenViewer(*image_shape)
        self.info = MemoryReader("SpeedRunners.exe")
        self.max_time = max_time
        self.image_shape = image_shape
        self.num_actions = self.actor.num_actions

        self.started = False
        self.state = []

        self.last_time = 0
        self.num_obstacles_hit = self.info.get_obstacles_hit()

        # The addresses of all the values to get
        self.addresses = {"x_speed" : 0x29CBF23C,
                          "y_speed" : 0x29CBF240,
                          "current_time" : 0x29C9584C,
                          "obstacles_hit" : 0x739D1334}

        # The offsets of all the values to get
        self.offsets = {"obstacles_hit" : [0x0, 0x0, 0x1F6C, 0x8, 0x4]}

    def reset(self):
        if(not self.started):
            self.sv.GetHWND("SpeedRunners")
            self.sv.Start()

            self.started = True

        self.actor.reset()

        self._get_new_state()

        return self.state

    def stop(self):
        self.reset()
        self.sv.Stop()
        self.info.close_handle()

    def step(self, action):
        self.actor.perform_action(action)

        self._get_new_state()
        state = self.state
        done, made_lap = self.episode_finished()
        reward = self._get_reward(made_lap)


        return state, reward, done

    def get_state(self):
        return self.state

    def _get_new_state(self):
        new_state = self.sv.GetScreen()

        while(np.array_equal(self.state, new_state)):
            new_state = self.sv.GetScreen()

        self.state = new_state

    def _get_reward(self, made_lap):
        reward = -1 / (self._get_speed() + 1)

        obst_dif = self._get_obstacles_hit() - self.num_obstacles_hit

        reward -= 0.05 * obst_dif
        self.num_obstacles_hit += obst_dif

        if(made_lap):
            reward += 10

        return reward

    def _get_x_speed(self):
        return self.info.get_address(self.addresses["x_speed"], c_float)

    def _get_y_speed(self):
        return self.info.get_address(self.addresses["y_speed"], c_float)

    def _get_speed(self):
        return np.sqrt(np.square(self.get_x_speed()) + 0.75 * np.square(self.get_y_speed()))

    def _get_current_time(self):
        return self.info.get_address(self.addresses["current_time"], c_float)

    def _get_obstacles_hit(self):
        return self.info.get_address(self.addresses["obstacles_hit"], ctypes.c_byte)

    def _episode_finished(self):
        finished = False

        if(self._get_current_time() > self.max_time
           win32api.GetAsyncKeyState(0x50)):
            self.last_time = 0
            finished = True
            made_lap = False
        elif(self._get_current_time() < self.last_time):
            self.last_time = 0
            finished = True
            made_lap = True
        else:
            self.last_time = self._get_current_time()
            finished = False
            made_lap = False

        return finished, made_lap

    def state_space(self):
        return self.image_shape

    def action_space(self):
        return self.num_actions
