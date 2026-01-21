# Premise 
Predictive Monitoring with Imprecise Sensors

Based on: 
- [1] "Runtime Monitoring for Markov Decision Processes" by Sebastian Junges, Hazem Torfah, and Sanjit A. Seshia, CAV 2021 

The code and explanations are to support experiments with the prototype. 
This project is hosted on [GitHub](https://github.com/monitoring-MDPs/premise). 

## Installation from source 

(Users of an artifact can skip these steps). 
- Install Storm with Python APIs in [the usual way](https://moves-rwth.github.io/stormpy/installation.html).
- Run `pip install -e .`

## Using a Docker container

We provide a docker container

## Experiments
We describe how to reproduce the experimental section of [1].

### How to run experiments?

First, be sure that `stats/` is empty (just to be sure). Then run
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
Optionally, to render these tables, run

```
cd util && pdflatex stats_main.tex
```

The file `stats_main.pdf` now contains the tables as in the paper. 
To inspect the pdf you must copy the pdf to your host system, using 
```
cp stats_main.pdf /data/
```
You can now open the pdf in your host system.


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


