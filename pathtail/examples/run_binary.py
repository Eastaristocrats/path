"""Run a small binary score update from command-line inputs."""

import argparse
import json

from pathtail import MixtureEProcess, binary_bound, binary_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", nargs=4, type=int, default=[3, 1, 1, 3])
    parser.add_argument("--outcome", type=int, default=2)
    parser.add_argument("--method", choices=["monomial", "pooled"], default="monomial")
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--normalization", choices=["tight", "conservative"], default="tight")
    args = parser.parse_args()
    try:
        bound = binary_bound(
            sum(args.counts),
            method=args.method,
            tolerance=args.tolerance,
            normalization=args.normalization,
        )
        pair = binary_pair(args.counts, args.outcome, method=args.method)
        audit = MixtureEProcess()
        log_e_value = audit.update_pair(pair, bound=bound, tolerance=args.tolerance)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "score": float(pair.score),
                "companion": float(pair.companion),
                "bound": bound,
                "log_e_value": log_e_value,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
