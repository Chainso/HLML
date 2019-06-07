import torch
import torch.multiprocessing as mp
import numpy as np

from copy import deepcopy

from .parallel_trainer import Trainer

class PopulationBasedTrainer(ParallelTrainer):
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
        ParallelTrainer.__init__(self, trainer, population_size)

        assert cutoff > 0
        assert population_size >= cutoff * 2
        assert 1 >= mutation_rate > 0

        self.cutoff = cutoff
        self.mutation_rate = mutation_rate
        self.perturb_weight = perturb_weight

    def evolve_population(self):
        """
        Replaces the bottom (cutoff) of the population with the top (cutoff) and
        performs a mutation
        """
        sortid = torch.argsort([trainer.score for trainer in self.trainers], -1)
        sortid = sortid.numpy()

        self.trainers = self.trainers[sortid]
        self.hyperparams = self.hyperparams[sortid]
        
        # Initialize population
        for i in range(self.cutoff):
            self.trainers[i] = deepcopy(self.trainers[self.num_trainers - i - 1])
            self.mutate_weights(self.trainers[i], self.hyperparams[i])
       
    def mutate_weights(self, trainer, hyperparams):
        """
        Mutates the hyperparameters with the mutation rate and perturb weight

        trainer : The trainer to mutate the hyperparameters for
        hyperparams : The hyperparameters to mutate
        """
        # Mutate the hyperparameters the set them again
        for hyperparam in hyperparams:
            param = self.hyperparams[hyperparam]
            if(str(self.param.__class__).find("Hyperparameter") != -1
                and param.search):
                perturb = torch.rand(1).item() > self.mutation_rate

                if(perturb):
                    perturb_amt = torch.rand(1).item() * 2 - 1
                    perturb_amt = perturb_amt * self.perturb_weight

                    # Add a small epsilon as well to get past 0
                    param.weights.data *= (1 + perturb_amt)
                    param.weights.data += 1e-8

            # Is an optimizer parameter
            else:
                param_group = param["param_group"]

                for optim_param in param_group:
                    if(str(optim_param.__class__).find("Hyperparameter")
                        != 1) and optim_param.search):
                        perturb = torch.rand(1).item() > self.mutation_rate

                        if(perturb):
                            perturb_amt = torch.rand(1).item() * 2 - 1
                            perturb_amt = perturb_amt * self.perturb_weight

                            # Add a small epsilon as well to get past 0
                            optim_param.weights.data *= (1 + perturb_amt)
                            optim_param.weights.data += 1e-8

        trainer.set_hyperparameters(hyperparams)

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
            ParallelTrainer.train(self, evolution_interval, save_path,
                                  save_interval, logs_path, *args)

            self.evolve_population()
