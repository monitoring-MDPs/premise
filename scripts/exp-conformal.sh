#!/bin/bash

set -x

if [ $# -lt 2 ]; then
    echo "Usage: $0 <stats_path> <model_path>"
    exit 1
fi

stats_path="$1"
model_path="$2"
log_path="$3"

n=10  # Number of repeats, default to 10 if not provided

if [ ! -d "$stats_path" ]; then
    echo "Error: stats_path '$stats_path' does not exist."
    exit 1
fi

if [ ! -d "$model_path" ]; then
    echo "Error: model_path '$model_path' does not exist."
    exit 1
fi

if [ ! -d "$log_path" ]; then
    echo "Error: log_path '$log_path' does not exist."
    exit 1
fi

for i in $(seq 1 $n); do
    python -m premise.interval.run_conformal_prediction 12h $stats_path -mc SnL-10x10 --dump-model $model_path --dump-stats $stats_path --no-target --run-id=$i &> $log_path/SnL-10x10-conf-pred-$i.log
    python -m premise.interval.run_conformal_prediction 12h $stats_path -mc airportA-7-10-10 --dump-model $model_path --dump-stats $stats_path --no-target --run-id=$i &> $log_path/airportA-7-10-10-conf-pred-$i.log
    python -m premise.interval.run_conformal_prediction 12h $stats_path -mc evadeV-5-3 --dump-model $model_path --dump-stats $stats_path --no-target --run-id=$i &> $log_path/evadeV-5-3-conf-pred-$i.log
    python -m premise.interval.run_conformal_prediction 12h $stats_path -mc evadeV-6-3 --dump-model $model_path --dump-stats $stats_path --no-target --run-id=$i &> $log_path/evadeV-6-3-conf-pred-$i.log
    python -m premise.interval.run_conformal_prediction 12h $stats_path -mc evadeI-15 --dump-model $model_path --dump-stats $stats_path --no-target --run-id=$i &> $log_path/evadeI-15-conf-pred-$i.log

    python -m premise.interval.run_conformal_prediction 12h $stats_path -mc SnLw-10x10 -sv pos --dump-model $model_path --dump-stats $stats_path --no-target --run-id=$i &> $log_path/SnLw-10x10-conf-pred-$i.log
    python -m premise.interval.run_conformal_prediction 12h $stats_path -mc evadeV-6-3-coarse -sv start turn c_ax c_ay c_dx c_dy --dump-model $model_path --dump-stats $stats_path --no-target --run-id=$i &> $log_path/evadeV-6-3-coarse-conf-pred-$i.log
    python -m premise.interval.run_conformal_prediction 12h $stats_path -mc airportA-7-10-10 -sv d p pobs turn --dump-model $model_path --dump-stats $stats_path --no-target --run-id=$i &> $log_path/airportA-7-10-10-coarse-conf-pred-$i.log
    python -m premise.interval.run_conformal_prediction 12h $stats_path -mc airportB-7-40-20 -sv d p pobs turn --dump-model $model_path --dump-stats $stats_path --no-target --run-id=$i &> $log_path/airportB-7-40-20-coarse-conf-pred-$i.log


    python -m premise.interval.run_conformal_prediction 12h $stats_path -mc airportA-7-40-20 --dump-model $model_path --dump-stats $stats_path --no-target --run-id=$i &> $log_path/airportA-7-40-20-conf-pred-$i.log
    python -m premise.interval.run_conformal_prediction 12h $stats_path -mc airportB-3-50-30 --dump-model $model_path --dump-stats $stats_path --no-target --run-id=$i &> $log_path/airportB-3-50-30-conf-pred-$i.log
    python -m premise.interval.run_conformal_prediction 12h $stats_path -mc airportB-7-40-20 --dump-model $model_path --dump-stats $stats_path --no-target --run-id=$i &> $log_path/airportB-7-40-20-conf-pred-$i.log
done
