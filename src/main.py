import application

import sys


def main():
    app = application.Application(
        name='Mini NPU Simulator'
    )

    app.run()


if __name__ == '__main__':
    sys.exit(main())
