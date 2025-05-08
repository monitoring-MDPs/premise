#!/bin/bash

now=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p out/logs
mkdir -p out/stats/$now
mkdir -p out/models/$now

sed "s/date/${now}/g" scripts/commands.sh | parallel --verbose --results out/logs/$now/{}/ --ungroup --eta --no-run-if-empty
