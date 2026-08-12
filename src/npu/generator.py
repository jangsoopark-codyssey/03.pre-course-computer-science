from npu import matrix


def generate_cross(size: int) -> matrix.Matrix:
    if size <= 0:
        raise ValueError('size must be greater than 0')

    data = [
        [0.0 for _ in range(size)]
        for _ in range(size)
    ]

    center = size // 2

    for i in range(size):
        data[center][i] = 1.0
        data[i][center] = 1.0

    return matrix.Matrix(data=data)


def generate_x(size: int) -> matrix.Matrix:
    if size <= 0:
        raise ValueError('size must be greater than 0')

    data = [
        [0.0 for _ in range(size)]
        for _ in range(size)
    ]

    for i in range(size):
        data[i][i] = 1.0
        data[i][size - i - 1] = 1.0

    return matrix.Matrix(data=data)
    