from torch.utils.tensorboard import SummaryWriter

from HLML.PyTorch.Trainers.trainer import Trainer

class SLTrainer(Trainer):
    def __init__(self, model, device, optimizer, optim_args, loss_function,
                 score_metric):
        """
        model : The model to train
        device : The device to train on
        """
        Trainer.__init__(self, model, device)

        self.optimizer = optimizer(self.model.parameters(), **optim_args)
 
        self.loss_function = loss_function
        self.score_metric = score_metric

        self.find_hyperparams()

    def train_batch(self, data, targets):
        """
        Trains the model for a single batch

        data : The data to train on
        targets : The targets of the model prediction
        """
        self.optimizer.zero_grad()

        prediction = self.model(data)

        loss = self.loss_function(prediction, targets)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()

    def train(self, epochs, data_loader, test_loader=None, save_path=None,
              save_interval=1, logs_path=None):
        """
        Trains the model for the number of epochs given

        epochs : The number of epochs to train for
        data_loader : The data loader to get the data and targets from
        test_loader : The data loader for the test set, will evaluate the test
                      set every epoch if given and a summary exists
        save_path : The path to save the model to, None if you do not wish to
                    save
        save_interval : If a save path is given, the number of epochs in between
                        each save
        logs_path : The path to save logs to
        """
        summary = SummaryWriter(logs_path) if logs_path is not None else None

        self.model.train()

        total_batches = 0

        for epoch in range(1, epochs + 1):
            avg_loss = 0
            num_batches = 0

            for data, targets in data_loader:
                loss = self.train_batch(data, targets)
 
                avg_loss += loss
                num_batches += 1
                total_batches += 1

                if(summary is not None):
                    summary.add_scalar("Loss Per Batch", loss, total_batches)


            avg_loss /= num_batches

            if(summary is not None):
                summary.add_scalar("Loss Per Epoch", avg_loss, epoch)

                if(test_loader is not None):
                    avg_score = 0
                    num_tests = 0

                    for data, targets in test_loader:
                        avg_score += self.eval(data, targets)
                        num_tests += 1

                    avg_score /= num_tests

                    summary.add_scalar("Test Score Per Epoch", avg_score, epoch)

            if(save_path is not None and epoch % save_interval == 0):
                self.save(save_path)

    def eval(self, data, targets):
        """
        Evaluates the model on the data
        """
        self.model.eval()

        prediction = self.model(data)
        score = self.score_metric(prediction, targets)

        return score.cpu().item()