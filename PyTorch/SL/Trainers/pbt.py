import torch.multiprocessing as mp

from copy import deepcopy

from .trainer import Trainer

class PopulationBasedTrainer(Trainer):
    def __init__(self, trainer, population_size, cutoff):
        """
        trainer : The trainer to augument with population based training
        population_size : The size of the population to train
        cutoff : The number of models that are replaced after each cycle
        """
        Trainer.__init__(self, None, trainer.device)

        assert population_size > 0
        assert cutoff > 0
        assert population_size > cutoff * 2

        self.population_size = population_size
        self.trainer = trainer
        self.cutoff = cutoff

        self.create_population()

    def _initialize_module(self, module):
        """
        Initializes the hyperparameters of a module

        module : The module to initialize the hyperparameters for
        """
        classname = module.__class__.__name__

        if classname.find('Hyperparameter') != -1:
            if(module.search):
                module.initialize()

    def create_population(self):
        """
        Creates a population of trainers
        """
        self.trainers = [trainer]

        # Initialize population
        for i in range(self.population_size - 1):
            pop_trainer = deepcopy(trainer)
            pop_trainer.model.apply(self._initialize_module)
            self.trainers.append(pop_trainer)

    def evolve_population(self):
        """
        Replaces the bottom (cutoff) of the population with the top (cutoff) and
        performs a mutation
        """
        pass

    def train_batch(self, *args):
        """
        Trains the model for a single batch
        """
        procs = []

        for trainer in self.trainers:
            proc = mp.Process(target = trainer.train_batch, args=args)
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()

    def train(self, evolution_interval, epochs, save_path=None, save_interval=1,
              *args):
        """
        Trains the model using the given trainer and population based training

        evolution_interval : The number of epochs in between the evolution of
                             a population
        epochs : The number of epochs to train for
        save_path : The path to save the model to, None if you do not wish to
                    save
        save_interval : If a save path is given, the number of epochs in between
                        each save
        args : Any additional arguments of the trainer that was wrapped
        """
        generations = epochs // evolution_interval

        for generation in range(1, generations + 1):
            procs = []

            # Train the models
            for trainer in self.trainers:
                proc = mp.Process(target = trainer.train,
                                  args=(evolution_interval, save_path,
                                        save_interval, *args))
                proc.start()
                procs.append(proc)

            for proc in procs:
                proc.join()

            procs = []

            # Then evaluation for evolution
            for trainer in self.trainers:
                proc = mp.Process(target = trainer.eval, args=args)
                proc.start()
                procs.append(proc)

            for proc in procs:
                proc.join()

            self.evolve_population()