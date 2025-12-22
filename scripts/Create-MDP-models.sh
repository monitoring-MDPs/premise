#!/bin/bash
set -x

uncertainty=0.2
vt=exact

# Load common DTMC benchmarks
prism_folder="premise/examples/common-dtmc"
out_folder="premise/examples/transformed-mdp"

brp_constants_list=("N=16,MAX=8" "N=32,MAX=9" "N=64,MAX=10")
# Expand brp_constants_list with U values from 0.01 to 0.50 in steps of 0.05
expanded=()
for base in "${brp_constants_list[@]}"; do
    for i in $(seq 10 5 50); do
        uval=$(printf "%.3f" "$(echo "$i/1000" | bc -l)")
        expanded+=("${base},PCHAN=${uval}")
    done
done
brp_constants_list=("${expanded[@]}")
brp_filename="brp.pm"
for constants in "${brp_constants_list[@]}"; do
    new_filename="brp-${constants//,/-}.drn"
    echo "Processing $brp_filename with constants $constants"
    if [ -f "$out_folder/$new_filename" ]; then
        echo "Output file already exists, skipping..."
        continue
    fi
    python premise/bn/add_uncertainty.py "$prism_folder/$brp_filename" "$out_folder/$new_filename" --type mdp --uncertainty $uncertainty --$vt --constants "$constants" --copy-labels
done

too_big=("crowds_10-5" "crowds_15-5" "crowds_20-5")
for file in $(ls -lS "$prism_folder"/crowds_*.pm | awk '{print $9}' | tac); do
    filename=$(basename "$file" .pm)
    if [[ " ${too_big[@]} " =~ " ${filename} " ]]; then
        echo "Skipping too big model $filename"
        continue
    fi
    echo "Processing $filename.pm"
    if [ -f "$out_folder/$filename.drn" ]; then
        echo "Output file already exists, skipping..."
        continue
    fi
    python premise/bn/add_uncertainty.py "$prism_folder/$filename.pm" "$out_folder/$filename.drn" --type mdp --uncertainty $uncertainty --$vt --copy-labels
done

# Lead wlan benchmarks
wlan_constants_list=("BOFF=6" "BOFF=10")
wlan_filename="wlan.nm"
for constants in "${wlan_constants_list[@]}"; do
    new_filename="wlan-${constants}.drn"
    echo "Processing $wlan_filename with constants $constants"
    if [ -f "./premise/examples/concrete-mdps/$new_filename" ]; then
        echo "Output file already exists, skipping..."
        continue
    fi
    ../storm-cond/build/bin/storm --prism ./premise/examples/baier-mdps/wlan.nm --constants "$constants" --exportbuild ./premise/examples/concrete-mdps/${new_filename} --build-all-labels
done

# Build Monitoring models
out_folder="premise/examples/monitoring-cond-mdps/"
trace_length=250
num_traces=25
if [ ! -f "$out_folder/airportA-7.drn" ]; then
    python premise/demo.py --unfolding --exact --model premise/examples/airportA-7.nm --constants "DMAX=500,PMAX=300" --risk "Pmax=? [F \"crash\"]" --trace-length $trace_length --number-traces $num_traces --seed 0 --create-benchmark $out_folder
fi
if [ ! -f "$out_folder/airportB-7.drn" ]; then
    python premise/demo.py --unfolding --exact --model premise/examples/airportB-7.nm --constants "DMAX=500,PMAX=300" --risk "Pmax=? [F \"crash\"]" --trace-length $trace_length --number-traces $num_traces --seed 0 --create-benchmark $out_folder
fi
if [ ! -f "$out_folder/hidden-incentive.drn" ]; then
    python premise/demo.py --unfolding --exact --model premise/examples/hidden-incentive.nm --constants "N=35" --risk "Pmax=? [F<=36 \"crash\"]" --trace-length $trace_length --number-traces $num_traces --seed 0 --create-benchmark $out_folder
fi
if [ ! -f "$out_folder/evade-monitoring.drn" ]; then
    python premise/demo.py --unfolding --exact --model premise/examples/evade-monitoring.nm --constants "N=20,RADIUS=20" --risk "Pmax=? [F<=75 \"crash\"]" --trace-length $trace_length --number-traces $num_traces --seed 0 --create-benchmark $out_folder
fi
if [ ! -f "$out_folder/refuelB.drn" ]; then
    python premise/demo.py --unfolding --exact --model premise/examples/refuelB.nm --constants "N=33,ENERGY=250" --risk "Pmax=? [F<=20 \"empty\"]" --trace-length $trace_length --number-traces $num_traces --seed 0 --create-benchmark $out_folder
fi


# Load BN benchmarks
jani_folder="premise/examples/BN-benchmarks-dtmc"
out_folder="premise/examples/BN-benchmarks-mdp"
bad_models=("sachs" "insurance" "andes" "pathfinder" "barley") # Models are not probabilistic

for file in $(ls -lS "$jani_folder"/*.jani | awk '{print $9}' | tac); do
    filename=$(basename "$file" .jani)
    if [[ " ${bad_models[@]} " =~ " ${filename} " ]]; then
        echo "Skipping bad model $filename"
        continue
    fi
    echo "Processing $filename.jani"
    if [ -f "$out_folder/$filename.drn" ]; then
        echo "Output file already exists, skipping..."
        continue
    fi
    python premise/bn/add_uncertainty.py "$file" "$out_folder/$filename.drn" --type mdp --uncertainty $uncertainty --$vt
done
