import torch
import cv2

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from operator import itemgetter

class DataProcessor:
    def __init__(self, data_path, image_size):
        self.data_path = data_path
        self.image_size = image_size
    
    def image_transformation(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def tensor_to_image(self, image):
        image = (image + 1) * (255.0 / 2)
        image = image.permute(0, 2, 3, 1)

        return image.detach().numpy()

    def get_dataset(self):
        dataset = datasets.ImageFolder(root = self.data_path,
                                       transform = self.image_transformation())

        return dataset

    def get_loader(self, batch_size, shuffle):
        return DataLoader(self.get_dataset(), batch_size = batch_size,
                          shuffle = shuffle)

    def get_samples(self, num_samples):
        data_loader = self.get_loader(num_samples, False)

        samples = None
        for data, _ in data_loader:
            samples = data
            break

        return samples

class GANRunner:
    def __init__(self, gan, data_proc, static_noise):
        self.gan = gan
        self.data_proc = data_proc
        self.static_noise = static_noise

    def train(self, test_func, model_save_path, model_test_path, epochs,
              batch_size, save_interval, disc_steps = 1):
        data_loader = self.data_proc.get_loader(batch_size, True)

        for epoch in range(epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            num_batches = 0

            for _, data in enumerate(data_loader, 0):
                data = data[0]

                gen_loss, disc_loss = self.gan.train_step(data, disc_steps)
                epoch_gen_loss += gen_loss
                epoch_disc_loss += disc_loss
                num_batches += 1

            epoch_gen_loss /= num_batches
            epoch_disc_loss /= num_batches

            print("Epoch " + str(epoch + 1) + ":", "Gen Loss", epoch_gen_loss,
                  "\tDisc Loss", epoch_disc_loss)
    
            if((epoch + 1) % save_interval == 0):
                self.gan.save(model_save_path)
                self.test(test_func, model_test_path, epoch + 1)

    def test(self, test_func, model_test_path, epoch = 1):
        gan_images = test_func(self.static_noise)
        gan_images = self.data_proc.tensor_to_image(gan_images)
        print()
        for i in range(len(gan_images)):
            cv2.imwrite(model_test_path + "/gan-" + str(epoch) + "-"
                        + str(i + 1) + ".jpg", gan_images[i])
