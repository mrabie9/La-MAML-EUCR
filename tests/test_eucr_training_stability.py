"""EUCR training should stay finite under Adam + short tuning-style schedules."""

from __future__ import annotations

# ruff: noqa: E402

import math
import sys
from pathlib import Path

import torch
from torch.autograd import Variable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parser as file_parser
from model.eucr import Net


def test_eucr_observe_stays_finite_for_five_epochs() -> None:
    """Two observe epochs must not produce NaN loss on the HPT tuning dataset."""
    if not torch.cuda.is_available():
        return

    args = file_parser.parse_args_from_yaml(
        [
            str(ROOT / "configs/tuning_defaults.yaml"),
            str(ROOT / "configs/models/til/eucr.yaml"),
        ]
    )
    loader_mod = __import__(
        "dataloaders.task_incremental_loader", fromlist=["IncrementalLoader"]
    )
    loader = loader_mod.IncrementalLoader(args, seed=0)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()
    model = Net(n_inputs, n_outputs, n_tasks, args).cuda()
    _, train_loader, _, _ = loader.new_task()

    for epoch in range(5):
        model.real_epoch = epoch
        for batch_x, batch_y in train_loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _, _ = model.observe(Variable(batch_x.cuda()), batch_y.cuda(), 0)
            assert math.isfinite(loss), f"loss became non-finite at epoch={epoch}"
