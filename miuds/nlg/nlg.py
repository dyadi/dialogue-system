from abc import ABCMeta, abstractmethod


class NLG(metaclass=ABCMeta):
    """
    NL Generation Interface
    Usage:

        >>> nlg = NLG(*args, **kwargs)
        >>> nlg([{'intent': 'greeting'}])
        'Hello!'
        >>> nlg([{'intent': 'inform', 'slot': 'price', 'filler': '$10'}])
        'It cost $10.'

    """
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def __call__(self, dialg_action):
        pass
