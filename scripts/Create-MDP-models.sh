#!/bin/bash
set -x

skip=$1
if [ -z "$skip" ]; then
    skip=0
fi


vt=exact

# Load common DTMC benchmarks
if [ "$skip" -le 0 ]; then
    prism_folder="premise/examples/common-dtmc"
    out_folder="premise/examples/transformed-mdp"

    brp_constants_list=("N=16,MAX=8,PCHAN=0.1" "N=32,MAX=8,PCHAN=0.1" "N=64,MAX=8,PCHAN=0.1"  "N=128,MAX=8,PCHAN=0.1"
                        "N=16,MAX=8,PCHAN=0.2" "N=32,MAX=8,PCHAN=0.2" "N=64,MAX=8,PCHAN=0.2"  "N=128,MAX=8,PCHAN=0.2")
    uncertainty_list=("0.05" "0.15")
    brp_filename="brp.pm"
    for uncertainty in "${uncertainty_list[@]}"; do
      for constants in "${brp_constants_list[@]}"; do
          new_filename="brp-${constants//,/-}-${uncertainty}.drn"
          echo "Processing $brp_filename with constants $constants"
          if [ -f "$out_folder/$new_filename" ]; then
              echo "Output file already exists, skipping..."
              continue
          fi
          python premise/bn/add_uncertainty.py "$prism_folder/$brp_filename" "$out_folder/$new_filename" --type mdp --uncertainty ${uncertainty} --$vt --constants "$constants" --copy-labels
      done
    done

    too_big=("crowds_20-5")
    uncertainty_list=("0.01" "0.05" "0.15")
    for uncertainty in "${uncertainty_list[@]}"; do
      for file in $(ls -lS "$prism_folder"/crowds_*.pm | awk '{print $9}' | tac); do
          filename=$(basename "$file" .pm)
          if [[ " ${too_big[@]} " =~ " ${filename} " ]]; then
              echo "Skipping too big model $filename"
              continue
          fi
          echo "Processing $filename.pm"
          if [ -f "$out_folder/$filename-${uncertainty}.drn" ]; then
              echo "Output file already exists, skipping..."
              continue
          fi
          python premise/bn/add_uncertainty.py "$prism_folder/$filename.pm" "$out_folder/$filename-${uncertainty}.drn" --type mdp --uncertainty ${uncertainty} --$vt --copy-labels
      done
    done
fi
#
## Load wlan benchmarks
if [ "$skip" -le 1 ]; then
    coins_constants_list=("K=4" "K=8" "K=12")
    coins_files=("03" "04" "05")

    for coins in "${coins_files[@]}"; do
        coin_filename="coin.${coins}.prism"

        for constants in "${coins_constants_list[@]}"; do
            new_filename="coin-${coins}-${constants}.drn"
            echo "Processing $wlan_filename with constants $constants"
            if [ -f "./premise/examples/concrete-mdps/$new_filename" ]; then
                echo "Output file already exists, skipping..."
                continue
            fi
            ../storm/build/bin/storm --prism ./premise/examples/baier-mdps/${coin_filename} --constants "$constants" --exportbuild ./premise/examples/concrete-mdps/${new_filename} --build-all-labels
        done
    done

    wlan_constants_list=("BOFF=4" "BOFF=6" "BOFF=10")
    for constants in "${wlan_constants_list[@]}"; do
        new_filename="wlan-${constants}.drn"
        echo "Processing $wlan_filename with constants $constants"
        if [ -f "./premise/examples/concrete-mdps/$new_filename" ]; then
            echo "Output file already exists, skipping..."
            continue
        fi
        ../storm/build/bin/storm --prism ./premise/examples/baier-mdps/wlan.nm --constants "$constants" --exportbuild ./premise/examples/concrete-mdps/${new_filename} --build-all-labels
    done
fi
#
## Build Monitoring models
#if [ "$skip" -le 2 ]; then
#    out_folder="premise/examples/monitoring-cond-mdps/"
#    trace_length=200
#    num_traces=25
#    if [ ! -f "$out_folder/full-airportA-7.drn" ]; then
#        python premise/demo.py --unfolding --exact --model premise/examples/airportA-7.nm --constants "DMAX=500,PMAX=300" --risk "Pmax=? [F \"crash\"]" --trace-length $trace_length --number-traces $num_traces --seed 0 --create-benchmark $out_folder
#    fi
#    if [ ! -f "$out_folder/full-airportB-7.drn" ]; then
#        python premise/demo.py --unfolding --exact --model premise/examples/airportB-7.nm --constants "DMAX=500,PMAX=300" --risk "Pmax=? [F \"crash\"]" --trace-length $trace_length --number-traces $num_traces --seed 0 --create-benchmark $out_folder
#    fi
#    if [ ! -f "$out_folder/full-hidden-incentive.drn" ]; then
#        python premise/demo.py --unfolding --exact --model premise/examples/hidden-incentive.nm --constants "N=35" --risk "Pmax=? [F<=40 \"crash\"]" --trace-length $trace_length --number-traces $num_traces --seed 0 --create-benchmark $out_folder
#    fi
#    if [ ! -f "$out_folder/full-evade-monitoring.drn" ]; then
#        python premise/demo.py --unfolding --exact --model premise/examples/evade-monitoring.nm --constants "N=20,RADIUS=20" --risk "Pmax=? [F<=80 \"crash\"]" --trace-length $trace_length --number-traces $num_traces --seed 0 --create-benchmark $out_folder
#    fi
#    if [ ! -f "$out_folder/full-refuelB.drn" ]; then
#        python premise/demo.py --unfolding --exact --model premise/examples/refuelB.nm --constants "N=33,ENERGY=250" --risk "Pmax=? [F<=50 \"empty\"]" --trace-length $trace_length --number-traces $num_traces --seed 0 --create-benchmark $out_folder
#    fi
#fi
#
## Load BN benchmarks
if [ "$skip" -le 3 ]; then
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
fi