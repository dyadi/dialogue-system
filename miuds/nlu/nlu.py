from abc import ABCMeta, abstractmethod


class NLU(metaclass=ABCMeta):
    """ NL Understanding Interface
    Usage:

        >>> nlu = NLG(*args, **kwargs)
        >>> nlu('Hello!')
        [{'intent': 'greeting'}]
        >>> nlu('It cost $10.')
        [{'intent': 'inform', 'slot': 'price', 'filler': '$10'}]

    """
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def __call__(self, text):
        pass
