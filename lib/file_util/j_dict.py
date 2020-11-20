from typing import Hashable


class j_dict(dict):
    """
        A wrapper around dict providing a few APIs that I think python should have provided.
    """

    def add_new_pair(self,
                     key: Hashable,
                     val: object) -> None:
        """
            Add the key and val pair to the dict. Throws exception if the key already exists in the dict.
        :param key: Key to be added.
        :param val: value corresponding to the key.
        :return: None
        """
        if key in self.keys():
            raise ValueError(f"The key: {key} already exists in this dict.")

        self.__setitem__(key, val)
