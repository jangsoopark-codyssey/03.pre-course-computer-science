from npu import matrix


class Simulator(object):

    def __init__(self, epsilon=1e-9):

        self._epsilon = epsilon

    def multiplication_accumulation(self, pattern: matrix.Matrix, filter_: matrix.Matrix) -> float:

        if pattern.shape != filter_.shape:
            raise ValueError(
                f'Matrix shape mismatch: '
                f'pattern={pattern.shape}, filter={filter_.shape}'
            )

        score = 0.0

        for row in range(pattern.rows):
            for col in range(pattern.cols):
                score += pattern[row][col] * filter_[row][col]

        return score

    def compare(self, score_a: float, score_b: float) -> str:
        if abs(score_a - score_b) < self._epsilon:
            return 'UNDECIDED'

        if score_a > score_b:
            return 'A'

        return 'B'