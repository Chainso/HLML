import torch
import torch.multiprocessing as mp
import numpy as np

from copy import deepcopy

from .trainer import Trainer

class PopulationBasedTrainer(Trainer):
    def __init__(self, trainer, population_size, cutoff, mutation_rate,
                 perturb_weight):
        """
        trainer : The trainer to augument with population based training
        population_size : The size of the population to train
        cutoff : The number of models that are replaced after each cycle
        mutation_rate : The chance of an individual hyperparameter being
                        perturbed
        perturb_weight : The range a hyperparameter can move
                         (perturb_weight * abs(hypermeter) 
        """
        Trainer.__init__(self, None, trainer.device)

        assert population_size > 0
        assert cutoff > 0
        assert population_size >= cutoff * 2
        assert 1 >= mutation_rate > 0

        self.population_size = population_size
        self.trainer = trainer
        self.cutoff = cutoff
        self.mutation_rate = mutation_rate
        self.perturb_weight = perturb_weight

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

        self.trainers = np.array(self.trainers)

    def evolve_population(self):
        """
        Replaces the bottom (cutoff) of the population with the top (cutoff) and
        performs a mutation
        """
        sortid = torch.argsort([trainer.score for trainer in self.trainers], -1)
        sortid = sortid.numpy()

        self.trainers = self.trainers[sortid]
        
        # Initialize population
        for i in range(self.cutoff):
            self.trainers[i].model = \
                deepcopy(self.trainers[self.population_size - i - 1].model)
            self.trainers[i].apply(self.mutate_hyperparameters)
       
    def mutate_weights(self, module):
        """
        Mutates the hyperparameters with the mutation rate and perturb weight

        module : The module to initialize the hyperparameters for
        """
        classname = module.__class__.__name__

        if classname.find('Hyperparameter') != -1:
            if(module.search):
                perturb = torch.rand(1).item() > self.mutation_rate

                if(perturb):
                    perturb_amt = torch.rand(1).item() * 2 - 1
                    perturb_amt = perturb_amt * self.perturb_weight
                    module.weights.data *= (1 + perturb_amt)
                
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
              logs_path=None, *args):
        """
        Trains the model using the given trainer and population based training

        evolution_interval : The number of epochs in between the evolution of
                             a population
        epochs : The number of epochs to train for
        save_path : The path to save the model to, None if you do not wish to
                    save
        save_interval : If a save path is given, the number of epochs in between
                        each save
        logs_path : The path to save logs to
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

            # Then evaluation for evolution
            self.eval()

            self.evolve_population()

    def eval(self, *args):
        """
        Evaluates the model on the data
        """
        procs = []

        for trainer in self.trainers:
                proc = mp.Process(target = trainer.eval, args=args)
                proc.start()
                procs.append(proc)

        for proc in procs:
            proc.join()

        scores = [trainer.score for trainer in self.trainers]
        self._score = np.max(scores)
        return scores