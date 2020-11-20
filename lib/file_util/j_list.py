import numpy as np


class j_list(list):
    """
        A normal list class with a few APIs renamed to make them C++ vector like,
        and with a few handful APIs added.
    """

    def push_back(self, item: object) -> None:
        self.append(item)

    def reset(self) -> None:
        self.clear()

    def to_np(self, datatype: str = None) -> np.array:
        return np.array(self) if datatype is None else np.array(self, datatype)
