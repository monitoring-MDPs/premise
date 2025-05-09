#!/bin/bash

now=$(date +"%Y-%m-%d_%H-%M-%S")

# Print it for debug
echo "Using timestamp: $now"

mkdir -p out/logs
mkdir -p out/stats/$now
mkdir -p out/models/$now

# Export so subshells can see it (not strictly needed, but clean)
export now

# Run commands with consistent timestamp
sed "s/date/${now}/g" scripts/commands-Antonina.sh | parallel \
  --verbose \
  --results out/logs/$now/{#}/ \
  --ungroup \
  --eta \
  --no-run-if-empty
