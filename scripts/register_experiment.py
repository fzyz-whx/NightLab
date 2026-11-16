"""Register a local classic ML run's aggregate snapshot; default is read-only."""
import argparse
import json
from pathlib import Path

from nightlab.services.local_experiments import register


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    try:
        result = register(Path(__file__).resolve().parents[1], args.run_name, args.execute)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'Experiment registration rejected: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
