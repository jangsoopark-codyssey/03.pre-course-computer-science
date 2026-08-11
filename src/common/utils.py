from typing import List, Dict, Any

import json



def input_matrix(name: str, rows: int, cols: int) -> List[List[float]]:
    print(f'{name} ({rows}줄 입력, 공백 구분)')

    data = []

    for _ in range(rows):
        while True:
            try:
                row = [
                    float(value)
                    for value in input().strip().split()
                ]
            except ValueError:
                print(
                    f'입력 형식 오류: 각 줄에 {cols}개의 숫자를 '
                    f'공백으로 구분해 입력하세요.'
                )
                continue

            if len(row) != cols:
                print(
                    f'입력 형식 오류: 각 줄에 {cols}개의 숫자를 '
                    f'공백으로 구분해 입력하세요.'
                )
                continue

            data.append(row)
            break

    return data


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def normalize_label(label: str):
    labels = {
        '+': 'Cross',
        'cross': 'Cross',
        'x': 'X',
    }

    return labels.get(str(label).strip().lower())
