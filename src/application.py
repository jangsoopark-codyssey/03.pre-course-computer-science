# application.py

from common import utils
from npu import matrix


class Application(object):

    def __init__(self, name, **params):
        self._name = name
        self._data_path = params.get('data_path')

        self._func = {
            '1': self.user_input_mode,
            '2': self.json_input_mode,
        }
        

    def menu(self):
        print(
            f'=== {self._name} ===\n\n'
            f'[모드 선택]\n\n'
            f'1. 사용자 입력 (3x3)\n'
            f'2. data.json 분석\n'
        )

        return input('선택: ').strip()

    def user_input_mode(self):
        
        print(
            "#----------------------------------------\n"
            "# [1] 필터 입력\n"
            "#---------------------------------------\n"
        )
        filter_a = matrix.Matrix(
            data=utils.input_matrix(name='필터 A', rows=3, cols=3)
        )

        print()

        filter_b = matrix.Matrix(
            data=utils.input_matrix(name='필터 B', rows=3, cols=3)
        )
        print(
            '\n'
            "#----------------------------------------\n"
            "# [2] 패턴 입력\n"
            "#---------------------------------------\n"
        )
        pattern = matrix.Matrix(
            utils.input_matrix(name='패턴', rows=3, cols=3)
        )

        print(filter_a)
        print(filter_b)
        print(pattern)


    def json_input_mode(self):
        data = utils.load_json(self._data_path)

        filters = data.get('filters', {})
        patterns = data.get('patterns', {})

        print(
            "#----------------------------------------\n"
            "# [1] 필터 로드\n"
            "#----------------------------------------"
        )

        for key in ('size_5', 'size_13', 'size_25'):
            filter_data = filters.get(key)

            if filter_data is None:
                print(f'✗ {key} 필터 로드 실패')
                continue

            if 'cross' not in filter_data or 'x' not in filter_data:
                print(f'✗ {key} 필터 스키마 오류')
                continue

            print(f'✓ {key} 필터 로드 완료 (Cross, X)')


    def run(self):
        
        choice = self.menu()

        mode = self._func.get(choice, None)
        if mode is None:
            return

        mode()
