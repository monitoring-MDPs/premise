#!/bin/bash
set -x

stats_path="$1"
model_path="$2"

if [ -z "$stats_path" ]; then
    echo "Usage: $0 <stats_path> <model_path>"
    exit 1
fi

if [ -z "$model_path" ]; then
    echo "Usage: $0 <stats_path> <model_path>"
    exit 1
fi

models=("airportB-7-40-20" "SnL-10x10" "airportA-7-10-10" "evadeV-5-3" "evadeV-6-3" "evadeI-15")

# RQ 1 analysis
python -m premise.interval.rq_1 -mc SnLw-10x10 -sv pos --stats-path "$stats_path"&
python -m premise.interval.rq_1 -mc evadeV-6-3-coarse -sv start turn c_ax c_ay c_dx c_dy --stats-path "$stats_path"&
python -m premise.interval.rq_1 -mc airportA-7-10-10 -sv d p pobs turn --stats-path "$stats_path"&
python -m premise.interval.rq_1 -mc airportB-7-40-20 -sv d p pobs turn --stats-path "$stats_path"&

for model in "${models[@]}"; do
    python -m premise.interval.rq_1 --mc "$model" --stats-path "$stats_path"&
done

wait

# RQ 2 analysis
python -m premise.interval.rq_2 -mc SnLw-10x10 -sv pos --stats-path "$stats_path"&
python -m premise.interval.rq_2 -mc evadeV-6-3-coarse -sv start turn c_ax c_ay c_dx c_dy --stats-path "$stats_path"&
python -m premise.interval.rq_2 -mc airportA-7-10-10 -sv d p pobs turn --stats-path "$stats_path"&
python -m premise.interval.rq_2 -mc airportB-7-40-20 -sv d p pobs turn --stats-path "$stats_path"&

for model in "${models[@]}"; do
    python -m premise.interval.rq_2 --mc "$model" --stats-path "$stats_path"&
done

wait


models=("SnL-10x10" "airportB-7-40-20" "airportA-7-10-10" "evadeV-5-3" "evadeV-6-3" "evadeI-15")
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
