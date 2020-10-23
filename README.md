# Premise (PREdictive Monitoring with Imprecise Sensors)

Based on: 
- [1] "Runtime Monitoring for Markov Decision Processes" by Sebastian Junges, Hazem Torfah, and Sanjit A. Seshia. We include a longer variant in paper.pdf. This paper additionally contains an outline of the proof of Theorem 3.

The code and explanations are to support experiments with the prototype. This is *not* a tool. 
This project is hosted on `https://github.com/monitoring-MDPs/premise` 

## Dependencies 

(Users of an artefact can skip these steps). 
- Install Storm with Python APIs in [the usual way](https://moves-rwth.github.io/stormpy/installation.html).
- Run `pip install tqdm pandas`

## How to run experiments?

Very simple: 
```
python experiments.py
```

We expect that this runs within ~3 hours (and using no more than 6 GB of RAM).
To select benchmarks, please edit experiments.py (in particular, the benchmarks array).
To speed up the computation, consider reducing the number of traces `--nr-traces X`.

Notice that running the experiments this creates a new folder in `stats/` for every benchmark. 
If such a folder already exists, the benchmark is skipped (irrespectively of the content of the folder). 
A warning is then printed.

## How to evaluate

Run:
```
python generate_tables.py stats
```

This creates texfiles for two tables `table1.tex` and `table2.tex`. 
To render these tables, run

```
cd util && pdflatex stats_main.tex
```

The file `stats_main.pdf` now contains the tables as in the paper. 
To recreate the original tables, please run  `python generate_tables.py paper_stats`.


## Algorithms

The actual algorithms have been integrated into the source code of [storm](https://www.stormchecker.org). Their entry points are:

- [Unfolding (header)](https://github.com/moves-rwth/storm/blob/master/src/storm-pomdp/transformer/ObservationTraceUnfolder.h) and [Unfolding (implementation)](https://github.com/moves-rwth/storm/blob/master/src/storm-pomdp/transformer/ObservationTraceUnfolder.cpp) 
- [Forward Filter (header)](https://github.com/moves-rwth/storm/blob/master/src/storm-pomdp/generator/NondeterministicBeliefTracker.h) and [Forward filter (implementation)](https://github.com/moves-rwth/storm/blob/master/src/storm-pomdp/generator/NondeterministicBeliefTracker.cpp)

## Source code

- `experiments.py` calls the `monitor` function in `demo.py` and writes data to a `stats` folder.
The source code clarifies the precise arguments and benchmarks we use. 
- `demo.py` contains a lightweight wrapper along the lines in [1, Fig. 5]: 
most of the ~200 lines of code are for logging statistics.
- `generate_tables.py` generates the Tables as in [1], based on the stats in `stats`

## Reference statistics

We have collected reference statistics that we used in the paper in `paper-stats`


