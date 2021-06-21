from abc import ABC, abstractmethod, ABCMeta
from typing import Dict
from state import State
import numpy as np


class Action(ABC):
    """The abstract base class definition of Action.
    
    An Action consists of two functions, 
    one is can_do(), which determines if the current state can perform the action, 
    and the other is do(), which performs the action on the current state.
    
    Extends:
        metaclass=ABCMeta
    """
    def __init__(self, priority: int = -1):
        self.priority = priority

    @abstractmethod
    def can_do(self, state: State):
        pass

    @abstractmethod
    def do(self, state: State):
        pass


if __name__ == '__main__':
    # test
    class InitAction(Action):

        def do(self, state, bsz, max_len, lens):
            state['stack'] = np.zeros((bsz, max_len), dtype='i2')
            print(lens.shape)
            assert lens.shape == (bsz, 1)
            state['buffer'] = np.tile(np.arange(max_len, dtype='i2'), (bsz, 1))
            state['buffer'][state['buffer']>lens] = 0
            state['buffer'][:, 0] = lens[:, 0]
            state['tree_a'] = np.zeros((bsz, max_len), dtype='i2')
            state['tree_r'] = np.zeros((bsz, max_len), dtype='i2')
            state['action'] = np.zeros((bsz, 1), dtype='i2')

        def can_do(self):
            pass

    init = InitAction()
    state = {}
    init.do(state, 2, 5, np.asarray([[3], [2]], dtype='i2'))
    print(state)