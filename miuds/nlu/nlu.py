from ABC import ABCMeta, abstractmethod

class NLU(metaclass=ABCMeta);
    """ NL Understanding Interface
    """
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def __call__(self, text):
        pass

