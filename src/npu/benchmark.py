# npu/benchmark.py

import time


class Benchmark(object):

    def __init__(self, iterations=10):
        assert iterations > 0

        self._iterations = iterations

    def measure(self, simulator, pattern, filter_) -> float:
        start = time.perf_counter()

        for _ in range(self._iterations):
            simulator.multiplication_accumulation(
                pattern=pattern,
                filter_=filter_
            )

        end = time.perf_counter()

        elapsed = end - start
        average = elapsed / self._iterations

        return average * 1000.0
