#!/bin/bash
set -x

n=${1:-10}  # Number of repeats, default to 10 if not provided

now=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p out/logs/$now
mkdir -p out/stats/$now
mkdir -p out/models/$now

# Generate repeated commands with run_id replaced
sed "s/date/${now}/g" scripts/commands.sh | sed '/^#/d' | \
while read -r cmd; do
    for i in $(seq 1 $n); do
        echo "$cmd" | sed "s/run_id/${i}/g"
    done
done | shuf | parallel --verbose --results out/logs/$now/{#}_{}/ --ungroup --eta --no-run-if-empty --joblog out/logs/$now/joblog.tsv --jobs 10

scripts/exp-conformal.sh out/stats/$now out/models/$now out/logs/$now
