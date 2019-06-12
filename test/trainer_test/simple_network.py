import torch
import torch.nn as nn
import numpy as np

from HLML.PyTorch.utils.classes import AMU

class Model(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        nn.Module.__init__(self)

        self.amu1 = AMU(num_inputs, 8, 3, num_hidden, 0.1)
        self.amu2 = AMU(num_hidden, 8, 3, num_outputs, 0.1)
        self.relu = nn.LeakyReLU(0.2)

        self.layers = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, inp):
        output, memory = self.amu1(inp)
        output = self.layers(output)
        output, memory = self.amu2(output, memory)
        output = output[:, -1, :]
        output = self.layers(output)

        return output

def get_data():
    data = torch.FloatTensor([[0, 0, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 1, 1],
                                [0, 1, 0, 0],
                                [0, 1, 0, 1],
                                [0, 1, 1, 0],
                                [0, 1, 1, 1],
                                [1, 0, 0, 0],
                                [1, 0, 0, 1],
                                [1, 0, 1, 0],
                                [1, 0, 1, 1],
                                [1, 1, 0, 0],
                                [1, 1, 0, 1],
                                [1, 1, 1, 0],
                                [1, 1, 1, 1]]).to(torch.device(device))
    data = data.view(4, 4, 4)
    targets = torch.FloatTensor([[0],
                                [1],
                                [1],
                                [0],
                                [1],
                                [0],
                                [0],
                                [1],
                                [1],
                                [0],
                                [0],
                                [1],
                                [0],
                                [1],
                                [1],
                                [0]]).to(torch.device(device))
    targets = targets[:4].view(4, 1)
    return data, targets

def accuracy_function(data, targets):
    return 1 - torch.abs(data - targets).mean()

if(__name__ == "__main__"):
    import os 
    import torch.multiprocessing as mp

    from torch.utils.data import TensorDataset, DataLoader

    from HLML.PyTorch.SL.sl_trainer import SLTrainer
    from HLML.PyTorch.Trainers.parallel_trainer import ParallelTrainer
    from HLML.PyTorch.utils.classes import Hyperparameter

    dir_path = os.path.dirname(os.path.realpath(__file__))

    batch_size = 1
    device = "cpu"

    train_dataset = TensorDataset(*get_data())
    eval_dataset = TensorDataset(*get_data())

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              pin_memory = device == "cuda")
    test_loader = DataLoader(eval_dataset, 16, shuffle=True,
                             pin_memory = device == "cuda")

    num_inputs = 4
    num_hidden = 16
    num_outputs = 1

    num_trainers = 1# mp.cpu_count()
    model = [Model(num_inputs, num_hidden, num_outputs)
             for i in range(num_trainers)]

    optimizer = torch.optim.SGD
    optim_params = {"lr" : Hyperparameter(1e-2, "lr"),
                    "momentum" : Hyperparameter(0, "momentum"),
                    "dampening": Hyperparameter(0, "dampening"),
                    "weight_decay" : Hyperparameter(0, "weight_decay")}

    loss_function = nn.BCELoss()
    score_metric = accuracy_function

    trainer = [SLTrainer(model[i], device, optimizer, optim_params, loss_function,
                        score_metric) for i in range(num_trainers)]
    trainer = ParallelTrainer(trainer, num_trainers)

    epochs = 100

    save_path = dir_path + "/models/test.torch"
    save_interval = 10
    logs_path = dir_path + "/logs"

    trainer.train(epochs, save_path, save_interval, logs_path, train_loader,
                  test_loader)