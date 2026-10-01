"""Run a label-only occupancy stopping example."""

import json

from pathtail import occupancy_pair


def main():
    record = occupancy_pair(
        iter([("left", 0.2), ("left", 0.4), ("right", 0.8)]),
        labels=["left", "right"],
        outcome_label="left",
        outcome_value=0.7,
        quota=1,
        cap=8,
    )
    print(
        json.dumps(
            {
                "score": record.pair.score,
                "companion": record.pair.companion,
                "draws": record.draws,
                "success": record.success,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
