import numpy as np
import win32api
import cv2
import os

from vizdoom import *

from HLML.envs import Env

class DoomEnv(Env):
    def __init__(self, max_steps, image_shape, scenario, wad, scale=1):
        Env.__init__(self)
        DoomGame.__init__(self)

        self.scenario = scenario
        self.wad = wad

        self._configure()

        self.image_shape = image_shape

        self.num_actions = len(self.get_available_buttons())

        self.scale = scale

        self.started = False

        self.old_state = None

    def _configure(self):
        # The path to vizdoom
        vizdoom_path = os.path.dirname(vizdoom.__file__)

        # Use CIG example config or your own.
        self.load_config(vizdoom_path + "/scenarios/" + self.scenario)
        
        self.set_doom_map("map01")  # Limited deathmatch.

        self.set_doom_scenario_path(vizdoom_path + "/scenarios/" + self.wad)
        
        # Name your agent and select color
        # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
        self.add_game_args("+name AI +colorset 0")
        
        # During the competition, async mode will be forced for all agents.
        self.set_mode(Mode.PLAYER)

        self.set_screen_resolution(ScreenResolution.RES_200X150)
        self.set_screen_format(ScreenFormat.GRAY8)

    def set_render(self, render):
        # Makes the window appear (turned on by default)
        self.set_window_visible(render)

    def reset(self):
        if(not self.started):
            self.init()

            self.started = True

        self.new_episode()
        return self._get_new_state()[0]

    def stop(self):
        self.close()

    def step(self, action):
        action_list = [0 for i in range(self.num_actions)]
        action_list[action] = 1

        reward = self.make_action(action_list)

        done = self.is_episode_finished()

        if(done):
            state = self.old_state
        else:
            state, mini_rew = self._get_new_state()
            reward += mini_rew

        return state, reward, done

    def _process_frame(self, frame):
        return cv2.resize(frame, self.image_shape[0:-1]) * self.scale

    def _get_new_state(self):
        reward = 0

        if(self.image_shape[-1] == 0):
            state = self._process_frame(self.get_state().screen_buffer)
        else:
            state = []

            i = 0
            while(i < self.image_shape[-1] and not self.is_episode_finished()):
                frame = self._process_frame(self.get_state().screen_buffer)

                state.append(frame)

                reward += self.make_action([0, 0, 1])
                i += 1

            for j in range(self.image_shape[-1] - len(state)):
                state.append(state[-1])

            state = np.stack(state, -1)

        self.old_state = state

        return state, reward

    def state_space(self):
        return self.image_shape

    def action_space(self):
        return self.num_actions
