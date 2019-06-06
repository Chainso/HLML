import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        nn.Module.__init__(self)

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(num_hidden, num_outputs),
            nn.Sigmoid()
        )

    def forward(self, inp):
        return self.layers(inp)

class Loader():
    def __init__(self, batch_size, device):
        self.batch_size = batch_size

        self.data = torch.FloatTensor([[0, 0, 0, 0],
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

        self.targets = torch.LongTensor([[0],
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

    def __iter__(self):
        zip_data = zip(self.data, self.targets)
        return enumerate(zip_data)

if(__name__ == "__main__"):
    from HLML.PyTorch.SL.sl_trainer import SLTrainer
    from HLML.PyTorch.utils.classes import Hyperparameter

    num_inputs = 4
    num_hidden = 16
    num_outputs = 1

    model = Model(num_inputs, num_hidden, num_outputs)
    device = "cuda"

    optimizer = torch.optim.SGD
    optim_params = {"lr" : Hyperparameter(1e-2, "lr"),
                    "momentum" : Hyperparameter(0, "momentum"),
                    "dampening": Hyperparameter(0, "dampening"),
                    "weight_decay" : Hyperparameter(0, "weight decay")}

    loss_function = nn.BCELoss()
    mse_loss = nn.MSELoss()
    score_metric = lambda x, t: torch.abs(x - t)

    trainer = SLTrainer(model, device, optimizer, optim_params, loss_function,
                        score_metric)

    epochs = 100
    data_loader = Loader(4, device)
    test_loader = Loader(4, device)
    save_path = "./models/test.torch"
    save_interval = 10
    logs_path = "./logs"

    trainer.train(epochs, data_loader, test_loader, save_path, save_interval,
                  logs_path)