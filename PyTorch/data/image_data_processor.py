import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ImageDataProcessor:
    """
    Processes images that are stored in a folder
    """
    def __init__(self, data_path, resized_image, device):
        """
        Creates the data processor using the images in the data path given as
        the data

        data_path : The path to the images
        resized_image : The size to resize the image to
        device : The device to store the images on, "cpu" for cpu, or "cuda"
                 for gpu
        """
        self.data_path = data_path
        self.resized_image = resized_image
        self.device = torch.device(device)

    def image_to_tensor(self):
        """
        Returns the pipeline of transformations that converts the image to a
        tensor. Resizes the image and normalizes to a [-1, 1] range
        """
        return transforms.Compose([
            transforms.Resize((self.resized_image, self.resized_image)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def tensor_to_image(self, image_tens):
        """
        Takes in the image as a tensor and returns back the PIL image

        image_tens : The tensor to convert to an image
        """
        image = (image_tens + 1) * (255.0 / 2)
        image = image.permute(0, 2, 3, 1)

        return image.detach().cpu().numpy()

    def get_dataset(self):
        """
        Returns the dataset specified by the datapath, transformed
        """
        dataset = datasets.ImageFolder(root = self.data_path,
                                       transform = self.image_to_tensor())

        return dataset

    def get_loader(self, batch_size, shuffle):
        """
        Returns the data loader to load the dataset

        batch_size : The batch size of the batches in the dataset
        shuffle : Will shuffle the dataset if true
        """
        pin_memory = str(self.device) == "cuda"

        return DataLoader(self.get_dataset(), batch_size = batch_size,
                          shuffle = shuffle, pin_memory = pin_memory)

    def get_samples(self, num_samples):
        """
        Will return the number of samples specified from the dataset

        num_samples : The number of samples to retrieve from the dataset
        """
        data_loader = self.get_loader(num_samples, False)

        samples = None
        for data, _ in data_loader:
            samples = data
            break

        return samples
