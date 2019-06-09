from abc import ABCMeta, abstractmethod


class Policy(metaclass=ABCMeta):
    '''
    Policy interface
    '''
    @abstractmethod
    def make_action(self, state):
        pass


class RLPolicy(Policy, metaclass=ABCMeta):
    '''
    Reinforcement Learning Policy
    '''
    def __init__(self):
        self._train_mode = False

    def set_train_mode(self):
        self._train_mode = True

    @abstractmethod
    def train_episode(self, *args, **kwargs):
        pass
