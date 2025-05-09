#!/bin/bash
set -x

now=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p out/logs
mkdir -p out/stats/$now
mkdir -p out/models/$now

sed "s/date/${now}/g" scripts/commands.sh | sed '/^#/d' | parallel --verbose --results out/logs/$now/{#}_{}/ --ungroup --eta --no-run-if-empty
