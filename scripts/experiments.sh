#!/bin/bash
set -x

n=${3:-10}  # Number of repeats, default to 10 if not provided

now=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p out/logs/$now
mkdir -p out/stats/$now
mkdir -p out/models/$now

jobs="$1"
if [ -z "$jobs" ]; then
    jobs=10  # Default number of jobs if not provided
fi

stats_path="$2"
if [ -z "$stats_path" ]; then
    stats_path="-"
fi


export OPENBLAS_NUM_THREADS=1

# Generate repeated commands with run_id replaced
sed "s/date/${now}/g" scripts/commands.sh | sed "s|stats_path|${stats_path}|g" | sed '/^#/d' | \
while read -r cmd; do
    for i in $(seq 1 $n); do
        echo "$cmd" | sed "s/run_id/${i}/g"
    done
done | shuf | parallel --verbose --results out/logs/$now/{#}_{}/ --ungroup --eta --no-run-if-empty --joblog out/logs/$now/joblog.tsv --jobs $jobs

scripts/exp-conformal.sh out/stats/$now out/models/$now out/logs/$now
