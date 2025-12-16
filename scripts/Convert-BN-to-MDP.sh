#!/bin/bash
# set -x

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
    if [ -f "$out_folder/$new_filename" ]; then
        echo "Output file already exists, skipping..."
        continue
    fi
    ../storm-cond/build/bin/storm --prism ./premise/examples/baier-mdps/wlan.nm --constants "$constants" --exportbuild ./premise/examples/concrete-mdps/wlan-${constants}.drn --build-all-labels
done

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
