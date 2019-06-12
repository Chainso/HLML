import gym
import numpy as np

from collections import deque
from gym import spaces

from HLML.envs import Env

class GymEnv(Env):
    def __init__(self, max_steps, name, render):
        Env.__init__(self, render)
        self.env = gym.make(name)

    def reset(self):
        state = self.env.reset()

        if(self.render):
            self.env.render()

        return state

    def step(self, action):
        state, reward, done, _ = self.env.step(action)

        if(not done and self.render):
            self.env.render()

        return state, reward, done

    def _get_new_state(self):
        pass

    def _get_reward(self):
        pass

    def episode_finished(self):
        pass

    def state_space(self):
        return [n for n in self.env.observation_space.shape]

    def action_space(self):
        return self.env.action_space.n

class StackedGymEnv(GymEnv):
    def __init__(self, max_steps, name, render, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        GymEnv.__init__(self, max_steps, name, render)
        self.env = FrameStack(self.env, k)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]