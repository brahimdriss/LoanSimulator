#!/usr/bin/env bash
# Aggregate every ablation campaign submitted by cluster/ablation.sub
# (mean/std CSVs across the 4 seeds, FL/social only). Login-node-safe: with
# CAMPAIGN and the knobs set per campaign, run_job.sh's aggregate pass finds
# every existing checkpoint and never trains or deploys anything. Each
# campaign's knob is re-exported for that pass on purpose -- if a checkpoint
# were missing, the pass would otherwise deploy it under the DEFAULT
# environment and silently mix a 1x run into an ablation campaign.
#
# Run when condor_q is empty:
#   cd ~/LoanSimulator && cluster/ablation_aggregate.sh
#
# Expect, per campaign/agent, one mean_fairness_lagrangian__social_*.csv and
# one std_*.csv under ~/eutopia_runs/<campaign>/<agent>/ with 3000 rows.
set -euo pipefail
cd "$(dirname "$0")/.."

unset WEIGHTS_CAMPAIGN LAMBDA_OVERRIDE FREEZE_LAMBDA SNAPSHOT_EPISODES \
      PERFORMATIVE_SCALE WEALTH_GAP_SCALE BUFFER_CAPACITY
export N_SEEDS=4 TRAIN_EPISODES=1000 DEPLOY_EPISODES=3000
export REWARD_FILTER=fairness_lagrangian CONSTRAINT_FILTER=social

# campaign | agents | knob assignment for that campaign
while IFS='|' read -r CAMP AGENTS KNOB; do
  for A in $AGENTS; do
    echo "=================== $CAMP / $A  ($KNOB) ==================="
    env CAMPAIGN="$CAMP" $KNOB cluster/run_job.sh "$A" aggregate
  done
done <<'EOF'
ablation_perf05|pepg pg|PERFORMATIVE_SCALE=0.5
ablation_perf2|pepg pg|PERFORMATIVE_SCALE=2
ablation_gap05|pepg pg|WEALTH_GAP_SCALE=0.5
ablation_gap2|pepg pg|WEALTH_GAP_SCALE=2
ablation_buf10|pepg|BUFFER_CAPACITY=10
ablation_buf200|pepg|BUFFER_CAPACITY=200
EOF

echo
echo "Aggregated CSVs (expect 3000-row mean_/std_ per campaign/agent):"
for CAMP in ablation_perf05 ablation_perf2 ablation_gap05 ablation_gap2 ablation_buf10 ablation_buf200; do
  for A in pepg pg; do
    d="$HOME/eutopia_runs/$CAMP/$A"
    [ -d "$d" ] || continue
    for f in "$d"/mean_fairness_lagrangian__social_*.csv; do
      [ -e "$f" ] && printf "  %-16s %-5s %s  %s rows\n" "$CAMP" "$A" "$(basename "$f")" "$(($(wc -l < "$f") - 1))"
    done
  done
done
