# Premise 
Predictive Monitoring with Imprecise Sensors

Based on: 
- [1] "Runtime Monitoring for Markov Decision Processes" by Sebastian Junges, Hazem Torfah, and Sanjit A. Seshia, CAV 2021 

The code and explanations are to support experiments with the prototype. 
This project is hosted on [GitHub](https://github.com/monitoring-MDPs/premise). 

## Installation from source 

(Users of an artifact can skip these steps). 
- Install Storm with Python APIs in [the usual way](https://moves-rwth.github.io/stormpy/installation.html).
- Run `python setup.py install` or equivalently `pip install .`

## Using a Docker container

We provide a docker container

```
docker pull sjunges/premise:cav21
```

The container is based on an container for the probabilistic model checker as provided by the Storm developers, for details, 
see [this documentation](https://www.stormchecker.org/documentation/obtain-storm/docker.html).

The following command will run the docker container (for Windows platforms, please see the documentation from the storm website).
```
docker run --mount type=bind,source="$(pwd)",target=/data -w /opt/premise --rm -it --name premise sjunges/premise:cav21
```
Files that one copies into `/data` are available on the host system in the current working directory. 

You will see a prompt inside the docker container. 

## How to run a single model?

For filtering, run: 
```
python premise/demo.py --filtering --exact --name "testname" --model examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F \"crash\"]"
```

For unfolding, run: 
```
python premise/demo.py --unfolding --exact --name "testname" --model examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F \"crash\"]"
```

#### Input
This will run filtering on the model `examples/airportA-3.nm` (with constants `DMAX=5,PMAX=5`). 
The risk is defined as the maximal probability of eventually crashing (in standard PRISM syntax).
The `testname` is used to identify the run in the created output. 

In particular, running this will simulate 50 traces of 500 steps each. 
These traces are each fed into a monitor that runs filtering. 

#### Output
The risk after every step, along with further statistics is written to `stats/testname-ff-ch-ea/` 

- `stats.out` contains some general model-dependent statistics.
- `...-SEED.csv` contains information for every trace. 
In particular, 
    - `Index` is the time step, 
    - `Observation` is an integer encoding the observed information, 
    - `Risk` is the actual risk. 

#### Options

Please run `python premise/demo.py -h` for a list of options.
 
To create reproducible results, one can fix the seed. You can also vary the number of traces or their length. 
- `--nr_traces 10` sets the number of traces to 10.
- `--trace-length 100` sets the lenght of a trace to 100.


## Experiments
We describe how to reproduce the experimental section of [1].

### How to run experiments?

Very simple. First, be sure that `stats/` is empty (just to be sure). Then run
```
python premise/experiments.py
```

We expect that this runs within ~3 hours (and using no more than 6 GB of RAM).
To select benchmarks, please edit experiments.py (in particular, the benchmarks array).
To speed up the computation, consider reducing the number of traces `--nr-traces X`.

Notice that running the experiments creates a new folder in `stats/` for every benchmark. 
If such a folder already exists, the benchmark is skipped (irrespectively of the content of the folder). 
A warning is then printed.

### How to evaluate the experiments?
While one can certainly manually evaluate all CSVs, we can automatically compile them into a table:
Run:
```
python premise/generate_tables.py stats
```

This creates texfiles for two tables `table1.tex` and `table2.tex`. 
To render these tables, run

```
cd util && pdflatex stats_main.tex
```

The file `stats_main.pdf` now contains the tables as in the paper. 


### Reference statistics

To recreate the original tables, please first run  `python premise/generate_tables.py paper_stats` (this will generate the right `tableX.tex`)

## Algorithms

The actual algorithms have been integrated into the source code of [storm](https://www.stormchecker.org). Their entry points are:

- [Unfolding (header)](https://github.com/moves-rwth/storm/blob/master/src/storm-pomdp/transformer/ObservationTraceUnfolder.h) and [Unfolding (implementation)](https://github.com/moves-rwth/storm/blob/master/src/storm-pomdp/transformer/ObservationTraceUnfolder.cpp) 
- [Forward Filter (header)](https://github.com/moves-rwth/storm/blob/master/src/storm-pomdp/generator/NondeterministicBeliefTracker.h) and [Forward filter (implementation)](https://github.com/moves-rwth/storm/blob/master/src/storm-pomdp/generator/NondeterministicBeliefTracker.cpp)

## Source code

In the premise folder, you can find the following sources.

- `monitoring.py` contains a lightweight wrapper along the lines in [1, Fig. 5]: 
most of the ~200 lines of code are for logging statistics.
- `demo.py` contains a command line interface to monitoring.py
- `experiments.py` calls the `monitor` function in `demo.py` and writes data to a `stats` folder.
The source code clarifies the precise arguments and benchmarks we use. 
- `generate_tables.py` generates the Tables as in [1], based on the stats in `stats`


