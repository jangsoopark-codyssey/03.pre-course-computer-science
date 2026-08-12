from typing import List, Tuple


class Matrix(object):

    def __init__(self, data: List[List[float]]):
        assert isinstance(data, list)
        assert len(data) > 0
        assert all(isinstance(row, list) for row in data)
        assert len(data[0]) > 0
        assert all(len(row) == len(data[0]) for row in data)

        self._data = data

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self._data), len(self._data[0])

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    def flatten(self):
        return [value for row in self._data for value in row]

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f'Matrix({self._data!r})'
