import torch
import torch.multiprocessing as mp
import numpy as np

from copy import deepcopy

from .trainer import Trainer

class ParallelTrainer(Trainer):
    def __init__(self, trainer, num_trainers=1):
        """
        A colletion of trainers to be trained in parallel

        trainer : Either a single trainer to make copies of, of a list of
                  trainers
        """
        Trainer.__init__(self, trainer.model, trainer.device)

        if(type(trainer) == list):
            self.trainers = trainer
        elif(issubclass(trainer, Trainer)):
            self.trainers = [trainer]

            for i in range(self.num_trainers - 1):
                self.trainers.append(deepcopy(trainer))
                self.trainers[-1].model.apply(self._initialize_module)

        self.hyperparams = []

        for trainer in self.trainers:
            self.hyperparams.append(trainer._find_hyperparams())

        self.trainers = np.array(self.trainers)
        self.hyperparams = np.array(self.hyperparams)

    def _initialize_module(self, module):
        """
        Initializes the hyperparameters of a module

        module : The module to initialize the hyperparameters for
        """
        classname = module.__class__.__name__

        if classname.find('Hyperparameter') != -1:
            if(module.search):
                module.initialize()

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

    def train(self, epochs, save_path=None, save_interval=1, logs_path=None,
              *args):
        """
        Trains the model using the given trainer group

        epochs : The number of epochs to train for
        save_path : The path to save the model to, None if you do not wish to
                    save
        save_interval : If a save path is given, the number of epochs in between
                        each save
        logs_path : The path to save logs to
        args : Any additional arguments of the trainer that was wrapped
        """
        procs = []

        # Train the models
        for trainer in self.trainers:
            proc = mp.Process(target = trainer.train,
                                args=(epochs, save_path, save_interval, *args))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()

        # Then evaluation for evolution
        self.eval()

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
        best_trainer = np.argmax(scores)

        self._score = scores[best_trainer]
        self._model = self.trainers[best_trainer].model

        return scores