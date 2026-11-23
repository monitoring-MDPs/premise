# Premise 
Predictive Monitoring with Imprecise Sensors

Based on: 
- [1] "Runtime Monitoring for Markov Decision Processes" by Sebastian Junges, Hazem Torfah, and Sanjit A. Seshia, CAV 2021 

The code and explanations are to support experiments with the prototype. 
This project is hosted on [GitHub](https://github.com/monitoring-MDPs/premise). 

## Installation from source

Premise uses [uv](https://docs.astral.sh/uv/) for its dependencies. Storm and its Python bindings are installed
automatically from the prebuilt [stormpy](https://pypi.org/project/stormpy/) wheels (version 2.0 or newer).

```
uv sync
```

This creates a virtual environment in `.venv` with premise installed in editable mode.
Commands can be run in this environment with `uv run`.

## Using a Docker container

Build the container with
```
docker build -t premise .
```

The following command runs the container and mounts the current directory at `/data`.
```
docker run --mount type=bind,source="$(pwd)",target=/data -it --name premise premise
```
Files that one copies into `/data` are available on the host system in the current working directory.

You will see a prompt inside the docker container, in which premise is installed.
If you want to run the container again after exiting it, you can start it again using `docker start premise -ai`.

## How to run a single model?

The example models can be found in the `examples` folder.

For filtering, run:
```
uv run python -m premise.demo --filtering --exact --name "testname" --model examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F \"crash\"]"
```

For unfolding, run:
```
uv run python -m premise.demo --unfolding --exact --name "testname" --model examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F \"crash\"]"
```

#### Input
This will run filtering/unfolding on the model `examples/airportA-3.nm` (with constants `DMAX=5,PMAX=5` defining the airport dimensions).
The risk is defined as the maximal probability of eventually crashing (in standard PRISM syntax).
The `testname` is used to identify the run in the created output.

In particular, running this will simulate 50 traces of 500 steps each.
These traces are each fed into a monitor that runs filtering.

#### Output
The risk after every step, along with further statistics is written to `stats/testname-ff-ch-ea/`

- `stats.out` contains some general model-dependent statistics.
- `stats-...-<seed>.csv` (filtering) or `trace-...-<seed>.csv` (unfolding) contains information for every trace.
In particular, for filtering,
    - `Index` is the time step,
    - `Observation` is an integer encoding the observed information,
    - `Risk` is the actual risk.

#### Options

Please run `uv run python -m premise.demo -h` for a list of options.

To create reproducible results, one can fix the seed with `--seed`. You can also vary the number of traces or their length.
- `--number-traces 10` sets the number of traces to 10.
- `--trace-length 100` sets the length of a trace to 100.


## Experiments

Run
```
uv run python -m premise.experiments --results-folder stats
```

To select benchmarks, please edit `premise/experiments.py` (in particular, the `benchmarks` array).
To speed up the computation, consider reducing the number of traces with `--number-traces X`,
or use `--fast` or `--smoke-test` to only run a subset of the configurations.

## Algorithms

The actual algorithms have been integrated into the source code of [storm](https://www.stormchecker.org). Their entry points are:

- [Unfolding (header)](https://github.com/moves-rwth/storm/blob/master/src/storm-pomdp/transformer/ObservationTraceUnfolder.h) and [Unfolding (implementation)](https://github.com/moves-rwth/storm/blob/master/src/storm-pomdp/transformer/ObservationTraceUnfolder.cpp) 
- [Forward Filter (header)](https://github.com/moves-rwth/storm/blob/master/src/storm-pomdp/generator/NondeterministicBeliefTracker.h) and [Forward filter (implementation)](https://github.com/moves-rwth/storm/blob/master/src/storm-pomdp/generator/NondeterministicBeliefTracker.cpp)

## Source code

In the premise folder, you can find the following sources.

- `monitoring.py` contains a lightweight wrapper along the lines in [1, Fig. 5]:
most of the code is for logging statistics.
- `demo.py` contains a command line interface to monitoring.py
- `experiments.py` calls `run_monitor` in `monitoring.py` for every benchmark and configuration and writes data to a results folder.
The source code clarifies the precise arguments and benchmarks we use.
- `monitor.py`, `oracle.py`, `trace_generator.py` and `traces.py` provide the interfaces used for learning (see `premise/__init__.py`).
