from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

class SynchronousEnvs():
    """
    Multiple instances of the same environment running synchronously
    """
    def __init__(self, env, *env_args, n_instances):
        """
        Creates the environments, all running synchronously

        env : The environment to create instances of
        env_args : The arguments for the environment
        n_instances : The number of instances to create
        """
        self.n_instances = n_instances

        self.envs = [env(*env_args) for _ in range(n_instances)]
        self.finished = [False for _ in range(len(self.envs))]

    def _parallel_funcs(self, funcs, func_args):
        """
        Executes the given functions in parallel and returns their result
        in order

        funcs : The functions to execute
        func_args : The list of arguments for each function
        """
        results = []

        with ProcessPoolExecutor(max_workers = cpu_count()) as executor:
            for func, args in zip(funcs, func_args):
                future = executor.submit(func, *args)
                future.add_done_callback(lambda x : results.append(x.result()))

        return results

    def _parallel_env_funcs(self, funcs, func_args):
        """
        Executes the given functions for the environments that are not finished
        their episode

        funcs : The functions to execute
        func_args : The list of arguments for each function
        """
        app_funcs = []
        app_func_args = []

        for i in range(self.n_instances):
            if(not self.finished[i]):
                pass
        return self._parallel_funcs(app_funcs, app_func_args)

    def reset(self):
        """
        Resets all instances of the environments
        """
        [env.reset() for env in self.envs]

    def step(self, actions):
        """
        Performs the given action in each instance of the environment

        actions : The actions for each instance of the environment
        """
        steps = []
        step_acts = []

        for i in range(self.envs):
            pass
        return self._apply_to_env(steps, step_acts)

    def episode_finished(self):
        """
        Returns true if the episode is finished for all instances of the
        environment
        """
        return all(env.episode_finished() for env in self.envs)

    def state_space(self):
        """
        Returns the dimensions of the environment state
        """
        return self.envs[0].state_space()

    def action_space(self):
        """
        Returns the number of actions for the environment
        """
        return self.envs[0].action_space()
