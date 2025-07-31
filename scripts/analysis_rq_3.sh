#!/bin/bash
set -x

stats_path="$1"
model_path="$2"

if [ -z "$stats_path" ] || [ -z "$model_path" ]; then
    echo "Usage: $0 <stats_path> <model_path>"
    exit 1
fi


models=("SnL-10x10" "airportA-7-10-10" "evadeV-5-3" "evadeV-6-3" "evadeI-15")
additional_args=("" "-ht")
additional_args_2=("" "-c")


# RQ 3 analysis
for ag_1 in "${additional_args[@]}"; do
    for ag_2 in "${additional_args_2[@]}"; do
        for model in "${models[@]}"; do
            python -m premise.interval.rq_3 --mc "$model" --stats-path "$stats_path" --model-path  "$model_path" $ag_1 $ag_2 &
        done
    done
done
