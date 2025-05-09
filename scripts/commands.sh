# Commands to run the experiments
# One command per line
# The string "date" will be replaced by the current date and time

# Normal refinement
timeount -v 4h python -u -m premise.interval.refinement -mc airportA-7-10-10 -ll 100 -ia 100 -ra 2 -d mse -ca 50 -cl 20 -ho 10 -sc threshold -sp 3 --dump-stats out/stats/date/airportA-7-10-10_refinement.npy -v --epsilon 0.005 -t -m out/models/date/airportA-7-10-10
timeount -v 4h python -u -m premise.interval.refinement -mc airportA-7-50-30 -ll 200 -ia 100 -ra 2 -d mse -ca 50 -cl 75 -ho 20 -sc threshold -sp 3 --dump-stats out/stats/date/airportA-7-50-30_refinement.npy -v --epsilon 0.0005 -t -m out/models/date/airportA-7-50-30
timeount -v 4h python -u -m premise.interval.refinement -mc airportB-7-50-30 -ll 200 -ia 100 -ra 2 -d mse -ca 50 -cl 75 -ho 20 -sc threshold -sp 3 --dump-stats out/stats/date/airportB-7-50-30_refinement.npy -v --epsilon 0.0005 -t -m out/models/date/airportB-7-50-30
timeount -v 4h python -u -m premise.interval.refinement -mc SnL-10x10 -ll 20 -ia 100 -ra 2 -t -ca 50 -cl 15 -ho 5 -sc threshold -sp 3 --dump-stats out/stats/date/SnL-10x10_refinement.npy -v -m out/models/date/SnL-10x10
timeount -v 4h python -m premise.interval.refinement -mc evadeV-6-3 -ll 60 -ia 100 -ra 2 -t -ca 50 -cl 40 -ho 12 -sc threshold -sp 3 -st 0.01 --dump-stats out/stats/date/evadev.npy -m out/models/date/evadeV -v
timeount -v 4h python -m premise.interval.refinement -mc evadeI-15 -ll 60 -ia 100 -ra 2 -t -ca 50 -cl 40 -ho 12 -sc threshold -sp 3 -st 0.01 --dump-stats out/stats/date/evadei.npy -m out/models/date/evadeI -v

# Compare methods
timeount -v 4h python -u -m premise.interval.run comp_methods -mc SnL-10x10 -ll 20 -ia 100 -ra 2 -t -ca 50 -cl 15 -ho 5 -sc threshold -sp 3 --dump-stats out/stats/date/SnL-10x10-comp-ref-stats.npy -v -m out/models/date/SnL-10x10-comp :: -mc SnL-10x10 -ll 20 -ia 100 -ra 2 -t -ca 50 -cl 15 -ho 5 --dump-stats out/stats/date/SnL-10x10-comp-noref-stats.npy -v -m out/models/date/SnL-10x10-comp-no-ref :: -mc SnL-10x10 -l 15 --horizon 5 -m out/models/date/SnL-10x10-comp-reg --dump-stats out/stats/date/SnL-10x10-comp-reg-stats.npy -a 0 -t 50

# Coarse models
timeount -v 4h python -m premise.interval.refinement -mc SnLw-10x10 -sv pos -ll 20 -ia 100 -ra 2 -t -ca 50 -cl 15 -ho 5 -sc threshold -sp 3 -st 0.01 --dump-stats out/stats/date/SnL-coarse.npy -m out/models/date/SnLC -v
timeount -v 4h python -m premise.interval.refinement -mc evadeV-6-3-coarse -sv start turn c_ax c_ay c_dx c_dy -ll 60 -ia 100 -ra 2 -t -ca 50 -cl 40 -ho 12 -sc threshold -sp 3 -st 0.01 --dump-stats out/stats/date/evadevcoarse.npy -m out/models/date/evadeVC -v
timeount -v 4h python -u -m premise.interval.refinement -mc airportB-7-50-30 -sv d p pobs turn -ll 200 -ia 100 -ra 2 -d mse -ca 50 -cl 75 -ho 20 -sc threshold -sp 3 --dump-stats out/stats/date/airportB-7-50-30-coarse_refinement.npy -v --epsilon 0.0005 -t -m out/models/date/airportB-7-50-30-coarse
