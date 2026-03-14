#!/bin/bash

# Auto-push script - commits and pushes changes every 10 minutes

if [[ -z "$1" ]]; then
    echo "Usage: $0 <machine-name>"
    exit 1
fi

MACHINE="$1"
REPO_DIR="/home/bdriss/LoanSimulator"

cd "$REPO_DIR" || exit 1

echo "Starting auto-push script for machine '$MACHINE' (every 10 minutes)"
echo "Press Ctrl+C to stop"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if there are any changes
    if [[ -n $(git status --porcelain) ]]; then
        echo "[$TIMESTAMP] Changes detected, pushing..."

        git pull --rebase
        git add -A
        git commit -m "Sync results ($MACHINE)"
        git push

        echo "[$TIMESTAMP] Push complete"
    else
        echo "[$TIMESTAMP] No changes to push"
    fi

    # Wait 10 minutes (600 seconds)
    sleep 600
done
