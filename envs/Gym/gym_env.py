import gym

from envs import Env

class GymEnv(Env):
    def __init__(self, max_steps, name, render):
        self.env = gym.make(name)
        self.render = render

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
