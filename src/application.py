from npu import generator
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

    def run(self):
        choice = self.menu()

        mode = self._func.get(choice, None)

        if mode is None:
            return

        mode()

    def user_input_mode(self):
        print(
            "#----------------------------------------\n"
            "# [1] 필터 입력\n"
            "#----------------------------------------\n"
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
            "# [2] 패턴 입력 및 생성\n"
            "#----------------------------------------\n"
        )

        input_pattern = matrix.Matrix(
            data=utils.input_matrix(
                name='사용자 입력 패턴',
                rows=3,
                cols=3
            )
        )

        patterns = {
            'User Input': input_pattern,
            'Generated Cross': generator.generate_cross(3),
            'Generated X': generator.generate_x(3),
        }

        print(
            '\n'
            "#----------------------------------------\n"
            "# [3] MAC 결과\n"
            "#----------------------------------------"
        )

        results = []

        for name, pattern in patterns.items():
            result = self._analyze_user_pattern(
                name=name,
                pattern=pattern,
                filter_a=filter_a,
                filter_b=filter_b
            )

            results.append(result)

        performances = self._user_performance_analysis(
            patterns=patterns,
            filter_a=filter_a
        )

        self._print_user_summary(
            results=results,
            performances=performances
        )

    def json_input_mode(self):
        data = utils.load_json(self._data_path)

        filters = self._load_filters(
            data.get('filters', {})
        )

        results = []

        print(
            '\n'
            "#----------------------------------------\n"
            "# [2] 패턴 분석 (라벨 정규화 적용)\n"
            "#----------------------------------------"
        )

        # JSON patterns
        for key, item in data.get('patterns', {}).items():
            result = self._analyze_pattern(
                key=key,
                item=item,
                filters=filters
            )

            results.append(result)

        # Generated Cross / X patterns
        generated_results = self._analyze_generated_patterns(
            filters=filters
        )

        results.extend(generated_results)

        performances = self._performance_analysis(
            filters=filters
        )

        self._print_summary(
            results=results,
            performances=performances
        )

    def _analyze_user_pattern(
        self,
        name,
        pattern,
        filter_a,
        filter_b
    ):
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

        average_time = self._benchmark.measure(
            func=self._npu.multiplication_accumulation,
            pattern=pattern,
            filter_=filter_a
        )

        print(f'\n--- {name} ---')
        print(f'A 점수: {score_a}')
        print(f'B 점수: {score_b}')
        print(
            f'연산 시간(평균/{self._num_iterations}회): '
            f'{average_time:.6f} ms'
        )
        print(f'판정: {result}')

        return (
            name,
            result,
            score_a,
            score_b,
            average_time
        )

    def _user_performance_analysis(
        self,
        patterns,
        filter_a
    ):
        print(
            '\n'
            "#----------------------------------------\n"
            f"# [4] 성능 분석 "
            f"(평균/{self._num_iterations}회)\n"
            "#----------------------------------------"
        )

        print(
            f'{"패턴":<20}'
            f'{"2D 시간(ms)":>18}'
            f'{"1D 시간(ms)":>18}'
            f'{"연산 횟수":>12}'
        )

        print('-' * 65)

        performances = []

        flat_filter = filter_a.flatten()

        for name, pattern in patterns.items():
            flat_pattern = pattern.flatten()

            time_2d = self._benchmark.measure(
                func=self._npu.multiplication_accumulation,
                pattern=pattern,
                filter_=filter_a
            )

            time_1d = self._benchmark.measure(
                func=self._npu.multiplication_accumulation_flat,
                pattern=flat_pattern,
                filter_=flat_filter
            )

            operations = (
                pattern.rows
                * pattern.cols
            )

            performances.append(
                (name, time_2d, time_1d)
            )

            print(
                f'{name:<20}'
                f'{time_2d:>18.6f}'
                f'{time_1d:>18.6f}'
                f'{operations:>12}'
            )

        return performances

    def _print_user_summary(
        self,
        results,
        performances
    ):
        print(
            '\n'
            "#----------------------------------------\n"
            "# [5] 결과 요약\n"
            "#----------------------------------------"
        )

        print(f'총 분석 패턴: {len(results)}개')

        print('\n판정 결과:')

        for (
            name,
            result,
            score_a,
            score_b,
            _
        ) in results:
            print(
                f'- {name}: '
                f'{result} '
                f'(A={score_a}, B={score_b})'
            )

        print('\n성능 비교:')

        for name, time_2d, time_1d in performances:
            self._print_performance_difference(
                name=name,
                time_2d=time_2d,
                time_1d=time_1d
            )

    def _load_filters(self, filters):
        loaded = {}

        print(
            "#----------------------------------------\n"
            "# [1] 필터 로드\n"
            "#----------------------------------------"
        )

        for key, item in filters.items():
            try:
                if 'cross' not in item or 'x' not in item:
                    raise ValueError(
                        '필터 스키마 오류'
                    )

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

            except (
                KeyError,
                TypeError,
                ValueError,
                AssertionError
            ) as e:
                print(
                    f'✗ {key} 필터 로드 실패 '
                    f'({e})'
                )

        return loaded

    def _analyze_pattern(
        self,
        key,
        item,
        filters
    ):
        try:
            _, size, _ = key.split('_')
            size = int(size)

            filter_ = filters[
                f'size_{size}'
            ]

            pattern = matrix.Matrix(
                data=item['input']
            )

            expected = utils.normalize_label(
                item['expected']
            )

            if expected is None:
                raise ValueError(
                    f'알 수 없는 expected 라벨: '
                    f'{item["expected"]}'
                )

            return self._analyze_json_pattern(
                key=key,
                pattern=pattern,
                expected=expected,
                filter_=filter_,
                size=size
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            AssertionError
        ) as e:
            print(f'\n--- {key} ---')
            print(f'FAIL ({e})')

            return (
                key,
                False,
                str(e)
            )

    def _analyze_generated_patterns(
        self,
        filters
    ):
        results = []

        print(
            '\n'
            "#----------------------------------------\n"
            "# [2-2] 자동 생성 패턴 분석\n"
            "#----------------------------------------"
        )

        for key, filter_ in filters.items():
            try:
                _, size = key.split('_')
                size = int(size)

                generated_patterns = {
                    f'{key}_generated_cross': (
                        generator.generate_cross(size),
                        'Cross'
                    ),
                    f'{key}_generated_x': (
                        generator.generate_x(size),
                        'X'
                    ),
                }

                for (
                    pattern_key,
                    pattern_data
                ) in generated_patterns.items():

                    pattern, expected = pattern_data

                    result = self._analyze_json_pattern(
                        key=pattern_key,
                        pattern=pattern,
                        expected=expected,
                        filter_=filter_,
                        size=size
                    )

                    results.append(result)

            except (
                KeyError,
                TypeError,
                ValueError,
                AssertionError
            ) as e:
                results.append(
                    (
                        key,
                        False,
                        str(e)
                    )
                )

        return results

    def _analyze_json_pattern(
        self,
        key,
        pattern,
        expected,
        filter_,
        size
    ):
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

        passed = (
            result == expected
        )

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
        print(
            f'Cross 점수: {score_cross}'
        )
        print(
            f'X 점수: {score_x}'
        )
        print(
            f'판정: {result} | '
            f'expected: {expected} | '
            f'{"PASS" if passed else "FAIL"}'
        )

        return (
            key,
            passed,
            reason
        )

    def _performance_analysis(
        self,
        filters
    ):
        print(
            '\n'
            "#----------------------------------------\n"
            f"# [3] 성능 분석 "
            f"(평균/{self._num_iterations}회)\n"
            "#----------------------------------------"
        )

        print(
            f'{"크기":<10}'
            f'{"2D 시간(ms)":>15}'
            f'{"1D 시간(ms)":>15}'
            f'{"연산 횟수":>15}'
        )

        print('-' * 55)

        performances = []

        for key, filter_ in filters.items():
            try:
                _, size = key.split('_')
                size = int(size)

                data = filter_['Cross']
                flat_data = data.flatten()

                time_2d = self._benchmark.measure(
                    func=self._npu.multiplication_accumulation,
                    pattern=data,
                    filter_=data
                )

                time_1d = self._benchmark.measure(
                    func=self._npu.multiplication_accumulation_flat,
                    pattern=flat_data,
                    filter_=flat_data
                )

                operations = (
                    size
                    * size
                )

                performances.append(
                    (
                        size,
                        time_2d,
                        time_1d
                    )
                )

                print(
                    f'{f"{size}x{size}":<10}'
                    f'{time_2d:>15.6f}'
                    f'{time_1d:>15.6f}'
                    f'{operations:>15}'
                )

            except (
                KeyError,
                TypeError,
                ValueError
            ):
                continue

        return performances

    def _print_summary(
        self,
        results,
        performances
    ):
        total = len(results)

        passed = sum(
            result[1]
            for result in results
        )

        failed = (
            total
            - passed
        )

        print(
            '\n'
            "#----------------------------------------\n"
            "# [4] 결과 요약\n"
            "#----------------------------------------"
        )

        print(
            f'총 테스트: {total}개'
        )
        print(
            f'통과: {passed}개'
        )
        print(
            f'실패: {failed}개'
        )

        failures = [
            result
            for result in results
            if not result[1]
        ]

        if failures:
            print(
                '\n실패 케이스:'
            )

            for (
                key,
                _,
                reason
            ) in failures:
                print(
                    f'- {key}: {reason}'
                )

        print(
            '\n성능 비교:'
        )

        for (
            size,
            time_2d,
            time_1d
        ) in performances:
            self._print_performance_difference(
                name=f'{size}x{size}',
                time_2d=time_2d,
                time_1d=time_1d
            )

    def _print_performance_difference(
        self,
        name,
        time_2d,
        time_1d
    ):
        if time_1d < time_2d:
            difference = (
                (time_2d - time_1d)
                / time_2d
                * 100.0
            )

            print(
                f'- {name}: '
                f'1D가 2D보다 '
                f'{difference:.2f}% 빠름'
            )

        elif time_2d < time_1d:
            difference = (
                (time_1d - time_2d)
                / time_2d
                * 100.0
            )

            print(
                f'- {name}: '
                f'1D가 2D보다 '
                f'{difference:.2f}% 느림'
            )

        else:
            print(
                f'- {name}: '
                f'2D와 1D 성능 동일'
            )