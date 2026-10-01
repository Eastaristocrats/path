"""Run the rational Volume construction on a small design."""

import json

from pathtail import volume_pair


def main():
    pair = volume_pair(
        features=[[1, 0], [1, 1], [1, 2]],
        targets=[0, 1, 4],
        outcome_features=[1, 3],
        outcome_target=9,
    )
    print(json.dumps({"score": str(pair.score), "companion": str(pair.companion)}, indent=2))


if __name__ == "__main__":
    main()
