#!/usr/bin/env python3
"""
SAC counterpart of pg_adapt.py: the SAME two-phase pipeline (Phase 1 on
the static IncomeEnvironment, Phase 2 fine-tuning on the performative
TestingIncomeEnvironment, same combos / seeds / artefacts / aggregation),
with loan_simulator.sac_agent.SACAgent in place of PolicyGradientAgent.
No performative component -- like PG, it never differentiates through the
population's response.

Implemented by pointing pg_adapt at the SAC agent through the
EUTOPIA_PG_AGENT environment variable (read at pg_adapt import time, and
inherited by the multiprocessing workers on both fork and spawn), rather
than copying pg_adapt.py's ~800 lines. Everything else -- CLI flags,
weights/results layout (with an "sac_" prefix on the Phase-1 weights and
"sac_" on the deploy artefacts, so cluster/verify_run.py and
probe_policies.py find them with --agents sac), aggregate mode -- is
pg_adapt.py's, unchanged.

Usage: identical to pg_adapt.py, e.g.
  python sac_adapt.py --data adult.csv --seed-list 0 --reward fairness_lagrangian \
      --constraint social --train-episodes 1000 --deploy-episodes 3000 ...
or through cluster/run_job.sh with agent "sac".
"""
import os
import sys

os.environ["EUTOPIA_PG_AGENT"] = "sac"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pg_adapt  # noqa: E402  (reads EUTOPIA_PG_AGENT at import)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    pg_adapt.main()
