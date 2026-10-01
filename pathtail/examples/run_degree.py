"""Solve one of three target designs sharing the same nuisance matrix."""

import argparse
import json

from pathtail import primitive_degree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["a", "b", "union"], default="union")
    args = parser.parse_args()
    nuisance = [[1, 0, 0], [1, 1, 0], [1, 2, 0], [0, 0, 1], [0, 0, 1]]
    targets = {"a": [0, 0, 1, 0, 0], "b": [0, 0, 0, 0, 1], "union": [0, 0, 1, 0, 1]}
    print(json.dumps(primitive_degree(nuisance, targets[args.target]), indent=2))


if __name__ == "__main__":
    main()
