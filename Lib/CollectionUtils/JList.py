import numpy as np


from typing import Any

class JList(list):
    """
        A normal list class with a few APIs renamed to make them C++ vector like,
        and with a few handful APIs added.
    """

    def push_back(self, item: object) -> None:
        self.append(item)

    def reset(self) -> None:
        self.clear()

    def get_item(self, idx:int) -> Any:
        return self.__getitem__(idx)

    def first_n_items(self, idx, first_n) -> Any:
        item = self.get_item(idx)
        return item[:first_n]

    def mean_f(self) -> float:
        """
            Returns the mean of all the elements in this list in float.
        """
        sum = 0.
        length = self.__len__()
        for i in range(length):
            sum += float(self.__getitem__(i))
        return sum / length

    def to_np(self, datatype: str = None) -> np.array:
        return np.array(self) if datatype is None else np.array(self, datatype)
