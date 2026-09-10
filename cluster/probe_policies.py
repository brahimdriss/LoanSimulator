#!/usr/bin/env python3
"""
Probe every saved Beta policy net in a campaign for the two failure modes the
run10 post-mortem found in the WEIGHTS (not just the metrics):

  SAT   a head pre-activation sits near the clamp / softplus saturation
        (mean < -5 or > 7). Below ~-5 softplus'(x) = sigmoid(x) < 0.007 and
        the head is effectively stuck at the Beta(1, .) floor; above ~7 it is
        driving a near-deterministic approve/reject.
  FLAT  the policy's Beta mean alpha/(alpha+beta) has std < 0.01 across a
        synthetic batch that spans the whole observation range, i.e. the net
        ignores its inputs (the coin-flip / approve-all / reject-all basins
        all look like this).

Each .pt is loaded into loan_simulator.policy_net.BetaPolicyNet(12, 128) and
evaluated on one fixed synthetic observation batch (512 rows, numpy seed 0)
covering the ranges the environment actually produces.

Usage:
    python3 cluster/probe_policies.py --root ~/eutopia_runs/fixpilot
    python3 cluster/probe_policies.py --root ~/eutopia_runs/fixpilot --stage trained
    python3 cluster/probe_policies.py --root C:/Users/vedan/run10_full_eo_results --agents pg

Legacy (run10-era) state_dicts lack the 'obs_scale' buffer. They were trained
on UNSCALED inputs, so for those the buffer is set to ones and the line is
tagged 'legacy' -- the probe then evaluates the net exactly as it ran. New
nets carry obs_scale in the state_dict and are loaded strictly.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from loan_simulator.policy_net import BetaPolicyNet  # noqa: E402

SAT_LO, SAT_HI = -5.0, 7.0
FLAT_STD = 0.01
N_ROWS = 512
BATCH_SEED = 0

# Observation columns in the order produced by the environment:
#   X/100, S, mu_R/100, mu_B/100, lambda_R, lambda_B, rho_R, rho_B,
#   theta_approval_prob, theta_S, theta_X, loan_amount
OBS_RANGES = [
    ("X/100", 0.2, 6.0),
    ("S", 0.0, 1.0),                 # binary, sampled as {0, 1}
    ("mu_R/100", 0.4, 10.0),
    ("mu_B/100", 0.4, 10.0),
    ("lambda_R", 1.0, 15.0),
    ("lambda_B", 1.0, 15.0),
    ("rho_R", 0.02, 0.3),
    ("rho_B", 0.02, 0.3),
    ("theta_approval_prob", 0.05, 0.95),
    ("theta_S", -2.0, 2.0),
    ("theta_X", -2.0, 2.0),
    ("loan_amount", 10.0, 60.0),
]


def synthetic_batch(n=N_ROWS, seed=BATCH_SEED):
    rng = np.random.default_rng(seed)
    cols = []
    for name, lo, hi in OBS_RANGES:
        if name == "S":
            cols.append(rng.integers(0, 2, size=n).astype(np.float32))
        else:
            cols.append(rng.uniform(lo, hi, size=n).astype(np.float32))
    return torch.from_numpy(np.stack(cols, axis=1))


def find_files(root, agent, stage):
    """Yield (path, combo, seed) for every policy file of one agent."""
    if stage == "deployed":
        d = os.path.join(root, agent, "deploy_artifacts")
        pat = re.compile(rf"^{re.escape(agent)}_(.+)__seed(\d+)_deployed\.pt$")
    else:
        d = os.path.join(root, agent, "weights")
        # pepg: <agent>_<combo>__seed<N>.pt ; pg: <combo>__seed<N>.pt
        pat = re.compile(rf"^(?:{re.escape(agent)}_)?(.+)__seed(\d+)\.pt$")
    out = []
    for f in glob.glob(os.path.join(d, "*.pt")):
        m = pat.match(os.path.basename(f))
        if m:
            out.append((f, m.group(1), int(m.group(2))))
    return sorted(out, key=lambda t: (t[1], t[2]))


def load_net(path):
    """Returns (net, legacy). legacy=True when the state_dict has no obs_scale
    (run10-era, trained on unscaled inputs) -- obs_scale is then set to ones."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["policy_net_state_dict"] if isinstance(blob, dict) and "policy_net_state_dict" in blob else blob
    net = BetaPolicyNet(12, 128)
    legacy = "obs_scale" not in sd
    if legacy:
        res = net.load_state_dict(sd, strict=False)
        unexpected = list(res.unexpected_keys)
        missing = [k for k in res.missing_keys if k != "obs_scale"]
        if unexpected or missing:
            raise RuntimeError(f"unexpected={unexpected} missing={missing}")
        with torch.no_grad():
            net.obs_scale.fill_(1.0)
    else:
        net.load_state_dict(sd, strict=True)
    net.eval()
    return net, legacy


def probe(net, x):
    with torch.no_grad():
        a_pre, b_pre = net.preactivations(x)
        alpha, beta = net(x)
    a_pre, b_pre = a_pre.squeeze(-1), b_pre.squeeze(-1)
    mean = (alpha / (alpha + beta)).squeeze(-1)
    stats = {
        "a_mean": a_pre.mean().item(), "a_min": a_pre.min().item(), "a_max": a_pre.max().item(),
        "b_mean": b_pre.mean().item(), "b_min": b_pre.min().item(), "b_max": b_pre.max().item(),
        "p_mean": mean.mean().item(), "p_std": mean.std().item(),
    }
    flags = []
    if not (SAT_LO <= stats["a_mean"] <= SAT_HI) or not (SAT_LO <= stats["b_mean"] <= SAT_HI):
        flags.append("SAT")
    if stats["p_std"] < FLAT_STD:
        flags.append("FLAT")
    return stats, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="campaign root (contains <agent>/ dirs)")
    ap.add_argument("--agents", nargs="+", default=["pepg", "pg"])
    ap.add_argument("--stage", choices=["deployed", "trained"], default="deployed",
                    help="deployed: deploy_artifacts/*_deployed.pt; trained: weights/*.pt (Phase 1)")
    a = ap.parse_args()
    root = os.path.expanduser(a.root)

    x = synthetic_batch()
    print(f"Probing {a.stage} policies under {root}  "
          f"(batch: {x.shape[0]} synthetic obs, seed {BATCH_SEED}; "
          f"SAT if head pre-act mean <{SAT_LO} or >{SAT_HI}; FLAT if std(Beta mean) <{FLAT_STD})")
    hdr = (f"{'agent':5} {'combo':32} {'seed':>4}  "
           f"{'alpha_pre mean/min/max':>26}  {'beta_pre mean/min/max':>26}  "
           f"{'E[p] mean':>9} {'std':>7}  flags")
    summary = {}
    for agent in a.agents:
        files = find_files(root, agent, a.stage)
        print(f"\n=== {agent} ({len(files)} files) ===")
        if not files:
            print("  (none found)")
            continue
        print(hdr)
        counts = {"SAT": 0, "FLAT": 0, "n": 0, "err": 0}
        for path, combo, seed in files:
            try:
                net, legacy = load_net(path)
            except Exception as e:
                counts["err"] += 1
                print(f"{agent:5} {combo:32} {seed:4}  LOAD ERROR: {e}")
                continue
            st, flags = probe(net, x)
            counts["n"] += 1
            for fl in flags:
                counts[fl] += 1
            tag = " ".join(flags) + (" legacy" if legacy else "")
            print(f"{agent:5} {combo:32} {seed:4}  "
                  f"{st['a_mean']:8.2f}/{st['a_min']:8.2f}/{st['a_max']:8.2f}  "
                  f"{st['b_mean']:8.2f}/{st['b_min']:8.2f}/{st['b_max']:8.2f}  "
                  f"{st['p_mean']:9.3f} {st['p_std']:7.4f}  {tag}")
        summary[agent] = counts

    print("\n" + "=" * 62)
    any_bad = False
    for agent, c in summary.items():
        bad = c["SAT"] or c["FLAT"] or c["err"]
        any_bad |= bool(bad)
        print(f"{agent}: SAT={c['SAT']}  FLAT={c['FLAT']}  of {c['n']} policies"
              + (f"  ({c['err']} load errors)" if c["err"] else ""))
    if any_bad:
        print("WARNING: saturated / input-blind policies present -- inspect the flagged rows.")
        sys.exit(1)
    print("PASS: no SAT or FLAT policies.")
    sys.exit(0)


if __name__ == "__main__":
    main()
