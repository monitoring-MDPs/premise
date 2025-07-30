#!/bin/bash
set -x

stats_path="$1"
if [ -z "$stats_path" ]; then
    echo "Usage: $0 <stats_path>"
    exit 1
fi

models=("SnL-10x10" "airportA-7-10-10" "evadeV-5-3" "evadeV-6-3" "evadeI-15")
additional_args=("" "-ht")

# RQ 1 analysis
for ag in "${additional_args[@]}"; do
    for model in "${models[@]}"; do
        python -m premise.interval.rq_1 --mc "$model" --stats-path "$stats_path" $ag
    done

    python -m premise.interval.rq_1 -mc SnLw-10x10 -sv pos --stats-path "$stats_path" $ag
    python -m premise.interval.rq_1 -mc evadeV-6-3-coarse -sv start turn c_ax c_ay c_dx c_dy "$stats_path" $ag
    python -m premise.interval.rq_1 -mc airportA-7-10-10 -sv d p pobs turn --stats-path "$stats_path" $ag
    python -m premise.interval.rq_1 -mc airportB-7-40-20 -sv d p pobs turn --stats-path "$stats_path" $ag
done