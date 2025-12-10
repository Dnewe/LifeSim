import argparse
import os
import sys


def parse_arguments() -> argparse.Namespace:
    '''
    Parse command-line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="File path to .yaml config"
    )
    return parser.parse_args()


def check_arguments(args: argparse.Namespace) -> None:
    '''
    Check if arguments are valid.
    '''
    file_extension = (str)(os.path.basename(args.config)).split('.')[-1]
    if not os.path.isfile(args.config) or not file_extension == 'yaml':
        print(f"Error: File'{args.config}' does not exist or is invalid (must be .yaml).")
        sys.exit(1)