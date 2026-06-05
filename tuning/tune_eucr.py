#!/usr/bin/env python3
"""Hyperparameter tuning harness for the EUCR learner.

Usage:
    python tuning/tune_eucr.py --hierarchical

    python tuning/tune_eucr.py \\
        --config configs/tuning_defaults.yaml \\
        --config configs/models/til/eucr.yaml \\
        --hierarchical

When ``--config`` is omitted, defaults are ``configs/tuning_defaults.yaml`` and
``configs/models/til/eucr.yaml``. Do not pass ``configs/base.yaml`` for tuning.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from tuning.hyperparam_tuner import make_main
    from tuning.presets import TUNING_PRESETS
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from tuning.hyperparam_tuner import make_main
    from tuning.presets import TUNING_PRESETS

main = make_main(TUNING_PRESETS["eucr"])

if __name__ == "__main__":
    main()
