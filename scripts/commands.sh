# Commands to run the experiments
# One command per line
# The string "date" will be replaced by the current date and time

# Normal refinement
python -u -m premise.interval.refinement -mc airportA-7-50-30 -ll 200 -ia 100 -ra 2 -d mse -ca 50 -cl 75 -ho 75 -sc threshold -sp 3 --dump-stats out/stats/date/airportA-7-50-30_refinement.npy -v --epsilon 0.0005 -t -m out/models/date/airportA-7-50-30
python -u -m premise.interval.refinement -mc SnL-10x10 -ll 20 -ia 100 -ra 2 -t -ca 50 -cl 15 -ho 5 -sc threshold -sp 3 --dump-stats out/stats/date/SnL-10x10_refinement.npy -v -m out/models/date/SnL-10x10

# Compare methods
python -u -m premise.interval.run comp_methods -mc SnL-10x10 -ll 20 -ia 100 -ra 2 -t -ca 50 -cl 15 -ho 5 -sc threshold -sp 3 --dump-stats out/stats/date/SnL-10x10-comp-ref-stats.npy -v -m out/models/date/SnL-10x10-comp :: -mc SnL-10x10 -ll 20 -ia 100 -ra 2 -t -ca 50 -cl 15 -ho 5 --dump-stats out/stats/date/SnL-10x10-comp-noref-stats.npy -v -m out/models/date/SnL-10x10-comp-no-ref :: -mc SnL-10x10 -l 15 --horizon 5 -m out/models/date/SnL-10x10-comp-reg --dump-stats out/stats/date/SnL-10x10-comp-reg-stats.npy -a 0 -t 50

# Coarse models
python -m premise.interval.refinement -mc SnLw-10x10 -sv pos -ll 20 -ia 100 -ra 2 -t -ca 50 -cl 15 -ho 5 -sc threshold -sp 3 -st 0.05 --dump-stats out/others/ref-c.npy -m out/others/SnLw-c -v
python -m premise.interval.refinement -mc evadeV-6-3-coarse -sv start turn c_ax c_ay c_dx c_dy -ll 60 -ia 100 -ra 2 -t -ca 50 -cl 40 -ho 12 -sc threshold -sp 3 -st 0.05 --dump-stats out/others/ref-c.npy -m out/others/SnLw-c -v
