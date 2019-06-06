from abc import ABC, abstractmethod

class Dataset(ABC):
    """
    An abstract dataset
    """
    def __init__(self, device, cuda=False):
        """
        Creates a dataset

        device : The device to load the dataset on, either "cpu" for cpu or
                 "cuda" for gpu
        ret_cuda : If true, will return cuda tensors
        """
        self.device = device
        self.cuda = cuda

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self):
        pass
