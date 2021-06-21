from collections import deque
from typing import Union

import numpy as np


class Stack(deque):
    """The stack is a first-in, last-out data structure."""

    def __init__(self, *args):
        super().__init__(*args)

    def push(self, x):
        return self.append(x)

    def pop(self):
        return super().pop()

    def __len__(self):
        return super().__len__()

    def is_empty(self):
        return self.__len__() == 0

    def clear(self):
        return super().clear()

    def first(self):
        return super().__getitem__(-1)


class Buffer(deque):
    """The buffer is a first-in, first-out data structure."""

    def __init__(self, *args):
        super().__init__(*args)

    def push(self, x):
        return self.append(x)

    def pop(self):
        return super().popleft()

    def __len__(self):
        return super().__len__()

    def is_empty(self):
        return self.__len__() == 0

    def clear(self):
        return super().clear()

    def first(self):
        return super().__getitem__(0)


class ActionList(deque):
    """The action list is a only in but not out data structure."""

    def __init__(self, *args):
        super().__init__(*args)

    def push(self, x):
        return self.append(x)

    def __len__(self):
        return super().__len__()

    def is_empty(self):
        return self.__len__() == 0

    def clear(self):
        return super().clear()


class AdjMatrix:
    """An adjacency matrix is a data structure that stores graph (containing tree)."""

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.adj = np.full((n_nodes, n_nodes), -1)

    def add_edge(i: int, j: int, r: int):
        assert r >= 0, f"The edge {r} must be a non-negative number."
        self.adj[i,j] = r

    def get_edge(i: int, j: int):
        assert 0<=i<self.n_nodes and 0<=j<self.n_nodes, f"i:{i}, j:{j} must be in the range [0, {self.n_nodes})"
        return self.adj[i,j]


DataStruct = Union[Stack, Buffer, ActionList, AdjMatrix]


if __name__ == '__main__':
    main()
