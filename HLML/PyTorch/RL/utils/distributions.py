from abc import ABC, abstractmethod

class Distribution(ABC):
    """
    A probability distribution
    """
    def __init__(self):
        """
        Creates a probability distribution
        """

    @abstractmethod
    def entropy(self):
        """
        Returns the entropy of the distribution
        """
