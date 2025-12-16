# BN Benchmark Script

Simple benchmarking script for testing Storm model checking on Bayesian Network models converted to MDPs.

## Usage

### Basic Usage

Run all models with default methods (bisection and restart) using both exact and floating-point arithmetic:
```bash
python premise/bn/benchmark.py
```

### Run Specific Models

```bash
python premise/bn/benchmark.py --models survey cancer asia
```

### Specify Conditional Methods

```bash
python premise/bn/benchmark.py --methods bisection restart bisection-advanced pi
```

Available methods:
- `bisection` - Bisection method
- `bisection-advanced` - Advanced bisection method
- `restart` - Restart method
- `pi` - Policy iteration

### Arithmetic Modes

The script automatically runs experiments using **both** exact and floating-point arithmetic for comprehensive comparison. Each model is loaded twice (once per arithmetic mode) and all methods are tested on both versions.

## Model Properties

Properties are defined at the top of `benchmark.py`. To add or change properties, edit the `PROPERTIES` dictionary:

```python
PROPERTIES = {
    "alarm": 'Pmax=? [F "HREKG0" || F "CVP0"]',
    "survey": 'Pmax=? [F "A0" || F "T0"]',
    # Add more as needed
}
```

Currently defined models:
- alarm, asia, cancer, child, earthquake, hailfinder, hepar2, survey, win95pts

## Plotting Results

After running benchmarks with JSON export, visualize the results:

```bash
python premise/bn/plot_results.py benchmark_results.json --output plots/
```

This generates:
- **scatter_METHOD1_vs_METHOD2_exact.png** - Scatter plot comparing two methods for exact arithmetic (circles=quantitative, triangles=bounded)
- **scatter_METHOD1_vs_METHOD2_float.png** - Scatter plot comparing two methods for floating-point arithmetic (circles=quantitative, triangles=bounded)
- **arithmetic_comparison.png** - Scatter plot comparing exact vs floating-point arithmetic execution times
- **speedup_heatmap_quantitative.png** - Heatmap showing speedup of all method×arithmetic combinations relative to restart/exact for quantitative queries
- **speedup_heatmap_bounded.png** - Heatmap showing speedup of all method×arithmetic combinations relative to restart/exact for bounded queries

## Example Output

```bash
$ python premise/bn/benchmark.py --models survey cancer --methods bisection restart

Benchmarking 2 models with methods: ['bisection', 'restart']
Testing both exact and floating-point arithmetic

cancer:

  [EXACT]
    Loaded: 23 states, 80 transitions
    bisection (Pmax=?):
      value: 0.9999996482 (states=23, time=0.048s)
    bisection (Pmax>=0.5):
      value: 1.0 (states=23, time=0.001s)
    restart (Pmax=?):
      value: 0.9999996482 (states=23, time=0.002s)
    restart (Pmax>=0.5):
      value: 1.0 (states=23, time=0.001s)

  [FLOAT]
    Loaded: 23 states, 80 transitions
    bisection (Pmax=?):
      value: 0.9999995232 (states=23, time=0.001s)
    bisection (Pmax>=0.5):
      value: 1.0 (states=23, time=0.000s)
    restart (Pmax=?):
      value: 0.9999974106 (states=23, time=0.001s)
    restart (Pmax>=0.5):
      value: 1.0 (states=23, time=0.001s)

survey:

  [EXACT]
    Loaded: 22 states, 139 transitions
    bisection (Pmax=?):
      value: 0.7418796161 (states=22, time=0.128s)
    bisection (Pmax>=0.5):
      value: 1.0 (states=22, time=0.001s)
    restart (Pmax=?):
      value: 0.7418796161 (states=22, time=0.003s)
    restart (Pmax>=0.5):
      value: 1.0 (states=22, time=0.003s)

  [FLOAT]
    Loaded: 22 states, 139 transitions
    bisection (Pmax=?):
      value: 0.7418797016 (states=22, time=0.001s)
    bisection (Pmax>=0.5):
      value: 1.0 (states=22, time=0.001s)
    restart (Pmax=?):
      value: 0.7418795061 (states=22, time=0.001s)
    restart (Pmax>=0.5):
      value: 1.0 (states=22, time=0.001s)

Results saved to benchmark_results.json
```
