from abc import ABC, abstractmethod

class Env(ABC):
    """
    An abstract environment
    """
    def __init__(self, render):
        """
        Creates the abstract environment

        render : If the environment is to be rendered to the screen
        """
        self.render = render

    def set_render(self, render):
        """
        Sets the environment to render the screen if true

        render : If the environment is to be rendered to the screen
        """
        self.render = render

    @abstractmethod
    def reset(self):
        """
        Resets the environment to the start of the episode
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Applies the given action to the environment

        action : The action to be taken
        """
        pass

    @abstractmethod
    def episode_finished(self):
        """
        Returns true if the episode is finished
        """
        pass

    @abstractmethod
    def state_space(self):
        """
        Returns the dimensions of the environment state
        """
        pass

    @abstractmethod
    def action_space(self):
        """
        Returns the number of actions for the environment
        """
        pass
