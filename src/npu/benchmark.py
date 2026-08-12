import time


class Benchmark(object):

    def __init__(self, iterations=10):
        assert iterations > 0

        self._iterations = iterations

    def measure(self, func, **kwargs):
        start = time.perf_counter()

        for _ in range(self._iterations):
            func(**kwargs)

        end = time.perf_counter()

        return (
            (end - start)
            / self._iterations
            * 1000.0
        )