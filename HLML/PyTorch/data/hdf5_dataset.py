import torch
import h5py
import numpy as np

from .base_dataset import Dataset
from HLML.PyTorch.utils import to_device

class HDF5Dataset(Dataset):
    """
    A general dataset for numerical data files
    """
    def __init__(self, hdf5_data, data_keys, device, ret_cuda=False):
        """
        Creates the dataset for the data file and labels file given

        hdf5_data : The file path to the hdf5 data file
        data_keys : The keys to retrieve from the hdf5 file
        device : The device to load the dataset on, either "cpu" for cpu or
                 "cuda" for gpu
        ret_cuda : If true, will return cuda tensors
        """
        Dataset.__init__(device, ret_cuda)

        self.data = h5py.File(hdf5_data, "r+")

        self.batch = [to_device(torch.tensor(list(self.data[key])), device)
                      for key in data_keys]

    def __len__(self):
        return len(self.batch[0])

    def __getitem__(self, idx):
        if(self.ret_cuda and self.device != "cuda"):
            items = [item[idx].cuda() for item in self.batch]
        else:
            items = [item[idx] for item in self.batch]

        return items

    def __add__(self, other):
        new_dataset = self.copy()
        new_dataset.batch = self.batch + other.batch

        return new_dataset

    def shuffle(self):
        """
        Shuffles the data in the dataset
        """
        randomize = np.arange(len(self))
        np.random.shuffle(randomize)

        self.batch = [item[randomize] for item in self.batch]

    def get_data(self):
        """
        Returns the data in the dataset
        """
        if(self.ret_cuda and self.device != "cuda"):
            ret_data = [item.cuda() for item in self.batch]
        else:
            ret_data = self.batch

        return ret_data
