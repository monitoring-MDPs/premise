#!/bin/bash

now=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p out/logs
mkdir -p out/res/$now

sed "s/date/${now}/g" scripts/commands.sh | parallel --verbose --results out/logs/$now/{}/ --ungroup --eta
