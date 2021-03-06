import torch
import cv2

from HLML.PyTorch.Trainers.trainer import Trainer

class GANRunner(Trainer):
    """
    A class used to train and test any GAN network
    """
    def __init__(self, gan, data_proc, static_noise, device):
        """
        Will create the GAN runner using the given GAN and data processor

        gan : The GAN to run
        data_proc : The data processor for the dataset to run the GAN on
        static_noise : The static noise to test the GAN on periodically
        device : The device to run the GAN on
        """
        Trainer.__init__(self, gan, device)

        self.data_proc = data_proc
        self.static_noise = static_noise

    def train_batch(self, data, *training_args):
        """
        Trains the model for a single batch

        data : The data to train on
        training_args : Any additional arguments required for the model
        """
        return self.model.train_batch(data, *training_args)

    def train(self, epochs, model_save_path, model_test_path, batch_size,
              save_interval, *training_args):
        """
        Starts training the GAN on the dataset

        epochs : The number of epochs to train for
        model_save_path : The path to save the GAN model to
        model_test_path : The path to save testing images to
        batch_size : The batch_size of the dataset
        save_interval : The number of epochs in between each model save and
                        testing period
        training_args : Any additional arguments required for the training
                        function of the GAN
        """
        data_loader = self.data_proc.get_loader(batch_size, True)

        for epoch in range(1, epochs + 1):
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            num_batches = 0

            for _, data in enumerate(data_loader):
                data = data[0].to(self.device)

                gen_loss, disc_loss = self.train_batch(data, *training_args)
                epoch_gen_loss += gen_loss
                epoch_disc_loss += disc_loss
                num_batches += 1

            epoch_gen_loss /= num_batches
            epoch_disc_loss /= num_batches

            print("Epoch " + str(epoch) + ":", "Gen Loss", epoch_gen_loss,
                  "\tDisc Loss", epoch_disc_loss)
    
            if(epoch % save_interval == 0):
                self.model.save(model_save_path)
                self.test(model_test_path, epoch)

                # Set back to training mode since testing sets the gan to eval
                self.model.train()

    def test(self, model_test_path, epoch = 1):
        """
        Tests the GAN on the static noise

        model_test_path : The path to save the testing images to
        epoch : The current epoch if training
        """
        # Set to eval mode to ignore conditional layers
        self.model.eval()
        gan_images = self.model(self.static_noise)
        gan_images = self.data_proc.tensor_to_image(gan_images)

        if(model_test_path[-1] == "/"):
            model_test_path = model_test_path[:-1]

        for image in range(len(gan_images)):
            cv2.imwrite(model_test_path + "/gan-" + str(epoch) + "-"
                        + str(image + 1) + ".jpg", gan_images[image])
