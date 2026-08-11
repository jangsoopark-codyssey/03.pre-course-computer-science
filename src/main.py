from common import definitions
import application

import sys
import os


def main():

    data_path = os.path.join(
        definitions.project_root,
        'data',
        'data.json'
    )

    app = application.Application(
        name='Mini NPU Simulator',
        data_path=data_path,
    )

    app.run()


if __name__ == '__main__':
    sys.exit(main())
