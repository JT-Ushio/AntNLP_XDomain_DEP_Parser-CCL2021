from typing import Dict
from data_structure import DataStruct

class State:

    def __init__(self, structures: Dict[str, DataStruct]):
        self.structures = structures

    def __getitem__(self, key):
        return self.structures.__getitem__(key)

    