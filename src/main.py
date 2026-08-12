from common import definitions

import application

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='Mini NPU Simulator'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data.json',
        help='JSON data file name (default: data.json)'
    )

    parser.add_argument(
        '--num-iterations',
        type=int,
        default=10,
        help='max num iterations for performance analysis'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = os.path.join(
        definitions.project_root,
        'data',
        args.data
    )

    app = application.Application(
        name='Mini NPU Simulator',
        data_path=data_path,
        num_iterations=args.num_iterations,
    )

    app.run()


if __name__ == '__main__':
    sys.exit(main())