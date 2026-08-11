# application.py

from common import utils


class Application(object):

    def __init__(self, name, **params):
        self._name = name
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
        pass

    def json_input_mode(self):
        pass

    def run(self):
        
        choice = self.menu()

        mode = self._func.get(choice, None)
        if mode is None:
            return

        mode()
