#!/usr/bin/env python3
"""Select a reproducible pilot subset of SWE-bench training-candidate instances.

Part of the DSpark on-policy agentic-trajectory pipeline (issue #51). See
`apps/dspark-swebench-runner.yaml` for the namespace/RBAC and
`apps/dspark-swebench-controller-20260715.yaml` for the generation controller
that consumes the JSON file this script produces.

Pool definitions (verified 2026-07-15 against the actual HF datasets):
  - full pool:      princeton-nlp/SWE-bench, split "test"    (2,294 instances)
  - Lite subset:    princeton-nlp/SWE-bench_Lite, split "test" (300 instances)
  - Verified subset: princeton-nlp/SWE-bench_Verified, split "test" (500 instances)

Training-candidate pool = full pool minus Lite minus Verified instance_ids.
SWE-bench-Verified must NEVER be used for training data generation -- it is
reserved as a held-out eval set. This script only ever selects from the
candidate pool, and additionally writes out the excluded-id counts so that's
auditable.

Selection is deterministic: sort candidate instance_ids lexicographically,
shuffle with a fixed seed (42, matching mini-swe-agent's own convention for
--shuffle), then take the first N. Re-running this script reproduces the
exact same pilot list as long as the upstream datasets don't change.

Usage:
    python3 dspark-swebench-select-instances.py --count 75 --seed 42 \
        --output dspark-swebench-instances-pilot.json
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone

from datasets import load_dataset

FULL_DATASET = "princeton-nlp/SWE-bench"
LITE_DATASET = "princeton-nlp/SWE-bench_Lite"
VERIFIED_DATASET = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"


def load_instance_ids(dataset_name: str, split: str) -> set[str]:
    ds = load_dataset(dataset_name, split=split)
    return {row["instance_id"] for row in ds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=75, help="Pilot sample size")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (reproducibility)")
    parser.add_argument("--output", default="dspark-swebench-instances-pilot.json")
    args = parser.parse_args()

    print(f"Loading full pool: {FULL_DATASET} [{SPLIT}]...", file=sys.stderr)
    full_ids = load_instance_ids(FULL_DATASET, SPLIT)
    print(f"  {len(full_ids)} instances", file=sys.stderr)

    print(f"Loading Lite subset: {LITE_DATASET} [{SPLIT}]...", file=sys.stderr)
    lite_ids = load_instance_ids(LITE_DATASET, SPLIT)
    print(f"  {len(lite_ids)} instances", file=sys.stderr)

    print(f"Loading Verified subset (HELD OUT, never for training): {VERIFIED_DATASET} [{SPLIT}]...", file=sys.stderr)
    verified_ids = load_instance_ids(VERIFIED_DATASET, SPLIT)
    print(f"  {len(verified_ids)} instances", file=sys.stderr)

    excluded_ids = lite_ids | verified_ids
    candidate_ids = sorted(full_ids - excluded_ids)
    print(
        f"Candidate pool (full - Lite - Verified) = {len(candidate_ids)} instances "
        f"(excluded {len(excluded_ids)}: {len(lite_ids)} Lite + {len(verified_ids)} Verified, "
        f"overlap {len(lite_ids & verified_ids)})",
        file=sys.stderr,
    )

    rng = random.Random(args.seed)
    shuffled = candidate_ids.copy()
    rng.shuffle(shuffled)
    pilot_ids = shuffled[: args.count]

    output = {
        "description": (
            "Pilot instance list for DSpark on-policy SWE-bench trajectory generation "
            "(issue #51). Selected from the training-candidate pool (full pool minus "
            "SWE-bench_Lite minus SWE-bench_Verified). SWE-bench_Verified is a held-out "
            "eval set and MUST NOT be used to generate training data."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "full_pool_dataset": FULL_DATASET,
        "full_pool_split": SPLIT,
        "full_pool_size": len(full_ids),
        "excluded_lite_dataset": LITE_DATASET,
        "excluded_lite_size": len(lite_ids),
        "excluded_verified_dataset": VERIFIED_DATASET,
        "excluded_verified_size": len(verified_ids),
        "candidate_pool_size": len(candidate_ids),
        "seed": args.seed,
        "count": args.count,
        "instance_ids": pilot_ids,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(pilot_ids)} pilot instance_ids to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
