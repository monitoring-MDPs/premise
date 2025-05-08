#!/bin/bash
set -x

base_args="-t -l 250 --epsilon 0.00005 --min-trans-prob 0.01"
testing_args="-l 100 -ho 10 --no-target -s 100"
suo_args="-sam premise/carla/carla_samples/all_data_more.pkl"
output_dir="out/test-refine-HHH"
sample_prefix="premise/carla/carla_samples/HHH"
increment=50

for i in {0..5}; do
    additional_samples=""
    for j in $(seq 0 $((i - 1))); do
        additional_samples+=" ${sample_prefix}_$((j * increment))_$(((j + 1) * increment)).pkl"
    done
    python -m premise.interval.learningIMC $base_args $suo_args $additional_samples -a $((1298 + i * 50)) -m $output_dir/carla-$i
done

for i in {0..5}; do
    python -m premise.interval.testing -sam premise/carla/carla_samples/test_data.pkl -t $output_dir/carla-$i-interval.npy -i $output_dir/carla-$i-initial_interval.npy $testing_args --dump-stats $output_dir/carla-$i-testing.npy
done