from npu import simulator
from npu import benchmark
from npu import matrix

from common import utils


class Application(object):

    def __init__(self, name, **params):
        self._name = name

        self._data_path = params.get('data_path')
        self._num_iterations = params.get('num_iterations', 10)

        self._npu = simulator.Simulator(
            epsilon=1e-9
        )

        self._benchmark = benchmark.Benchmark(
            iterations=self._num_iterations
        )

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
            data=utils.input_matrix(
                name='필터 A',
                rows=3,
                cols=3
            )
        )

        print()

        filter_b = matrix.Matrix(
            data=utils.input_matrix(
                name='필터 B',
                rows=3,
                cols=3
            )
        )

        print(
            '\n'
            "#----------------------------------------\n"
            "# [2] 패턴 입력\n"
            "#---------------------------------------\n"
        )

        pattern = matrix.Matrix(
            data=utils.input_matrix(
                name='패턴',
                rows=3,
                cols=3
            )
        )

        # Simulation
        score_a = self._npu.multiplication_accumulation(
            pattern=pattern,
            filter_=filter_a
        )

        score_b = self._npu.multiplication_accumulation(
            pattern=pattern,
            filter_=filter_b
        )

        result = self._npu.compare(
            score_a=score_a,
            score_b=score_b
        )

        # Benchmark
        average_time = self._benchmark.measure(
            simulator=self._npu,
            pattern=pattern,
            filter_=filter_a
        )

        print(
            '\n'
            "#----------------------------------------\n"
            "# [3] MAC 결과\n"
            "#----------------------------------------"
        )

        print(f'A 점수: {score_a}')
        print(f'B 점수: {score_b}')
        print(
            f'연산 시간(평균/{self._num_iterations}회): '
            f'{average_time:.6f} ms'
        )
        print(f'판정: {result}')

    def json_input_mode(self):
        data = utils.load_json(self._data_path)

        filters = self._load_filters(
            data.get('filters', {})
        )

        results = []

        for key, item in data.get('patterns', {}).items():
            result = self._analyze_pattern(
                key=key,
                item=item,
                filters=filters
            )

            results.append(result)

        self._performance_analysis(filters)
        self._print_summary(results)

    def run(self):
        choice = self.menu()

        mode = self._func.get(choice, None)

        if mode is None:
            return

        mode()

    def _load_filters(self, filters):
        loaded = {}

        print(
            "#----------------------------------------\n"
            "# [1] 필터 로드\n"
            "#----------------------------------------"
        )

        for key, item in filters.items():
            if 'cross' not in item or 'x' not in item:
                print(f'✗ {key} 필터 스키마 오류')
                continue

            loaded[key] = {
                'Cross': matrix.Matrix(
                    data=item['cross']
                ),
                'X': matrix.Matrix(
                    data=item['x']
                ),
            }

            print(
                f'✓ {key} 필터 로드 완료 '
                f'(Cross, X)'
            )

        return loaded

    def _analyze_pattern(self, key, item, filters):
        try:
            _, size, _ = key.split('_')
            size = int(size)

            filter_ = filters[f'size_{size}']

            pattern = matrix.Matrix(
                data=item['input']
            )

            if pattern.shape != (size, size):
                raise ValueError(
                    '패턴 크기 불일치'
                )

            if pattern.shape != filter_['Cross'].shape:
                raise ValueError(
                    'Cross 필터 크기 불일치'
                )

            if pattern.shape != filter_['X'].shape:
                raise ValueError(
                    'X 필터 크기 불일치'
                )

            expected = utils.normalize_label(
                item['expected']
            )

            if expected is None:
                raise ValueError(
                    f'알 수 없는 expected 라벨: '
                    f'{item["expected"]}'
                )

            score_cross = (
                self._npu.multiplication_accumulation(
                    pattern=pattern,
                    filter_=filter_['Cross']
                )
            )

            score_x = (
                self._npu.multiplication_accumulation(
                    pattern=pattern,
                    filter_=filter_['X']
                )
            )

            result = self._npu.compare(
                score_a=score_cross,
                score_b=score_x
            )

            result = {
                'A': 'Cross',
                'B': 'X',
                'UNDECIDED': 'UNDECIDED',
            }[result]

            passed = result == expected

            reason = None

            if not passed:
                if result == 'UNDECIDED':
                    reason = (
                        '동점(UNDECIDED) 처리 규칙에 '
                        '따라 FAIL'
                    )
                else:
                    reason = (
                        f'판정 불일치 '
                        f'(result={result}, '
                        f'expected={expected})'
                    )

            print(f'\n--- {key} ---')
            print(f'Cross 점수: {score_cross}')
            print(f'X 점수: {score_x}')
            print(
                f'판정: {result} | '
                f'expected: {expected} | '
                f'{"PASS" if passed else "FAIL"}'
            )

            return key, passed, reason

        except (
            KeyError,
            ValueError,
            AssertionError
        ) as e:
            print(f'\n--- {key} ---')
            print(f'FAIL ({e})')

            return key, False, str(e)

    def _performance_analysis(self, filters):
        print(
            '\n'
            "#----------------------------------------\n"
            f"# [3] 성능 분석 "
            f"(평균/{self._num_iterations}회)\n"
            "#----------------------------------------"
        )

        print(
            f'{"크기":<10} '
            f'{"평균 시간(ms)":<18} '
            f'{"연산 횟수":<10}'
        )

        print('-' * 42)

        for size in (3, 5, 13, 25):
            key = f'size_{size}'

            filter_ = filters.get(key)

            if filter_ is None:
                print(
                    f'{size}x{size:<6} '
                    f'필터 없음'
                )
                continue

            data = filter_['Cross']

            average_time = self._benchmark.measure(
                simulator=self._npu,
                pattern=data,
                filter_=data
            )

            print(
                f'{size}x{size:<6} '
                f'{average_time:<18.6f} '
                f'{size * size}'
            )

    def _print_summary(self, results):
        total = len(results)

        passed = sum(
            result[1]
            for result in results
        )

        failed = total - passed

        print(
            '\n'
            "#----------------------------------------\n"
            "# [4] 결과 요약\n"
            "#----------------------------------------"
        )

        print(f'총 테스트: {total}개')
        print(f'통과: {passed}개')
        print(f'실패: {failed}개')

        failures = [
            result
            for result in results
            if not result[1]
        ]

        if failures:
            print('\n실패 케이스:')

            for key, _, reason in failures:
                print(
                    f'- {key}: {reason}'
                )
                