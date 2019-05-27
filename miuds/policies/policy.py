from abc import ABCMeta, abstractmethod

class Policy(metaclass=ABCMeta):
    '''
    Policy interface
    '''
    @abstractmethod
    def make_action(self, state):
        pass

