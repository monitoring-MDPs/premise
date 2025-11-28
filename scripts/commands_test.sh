# Commands to run the experiments
# One command per line
# The string "date" will be replaced by the current date and time

# Compare methods
# python -u -m premise.interval.run 6h comp_methods -acas 0.5 --run-id=run_id -ll 30 -t -cl 20 -ho 10 -st 0.005 -sp 3 --dump-stats out/stats/date/acas0.5-ref-stats-run_id.npy -v --epsilon 0.00001 -m out/models/date/acas0.5-comp-ref-run_id :: -acas 0.5 --run-id=run_id -ll 30 -t -cl 20 -ho 10 --dump-stats out/stats/date/acas0.5-comp-noref-stats-run_id.npy -v --epsilon 0.00001 -m out/models/date/acas0.5-comp-no-ref-run_id :: -acas 0.5 --run-id=run_id --l 20 --horizon 10 -m out/models/date/acas0.5-comp-reg-run_id --dump-stats out/stats/date/acas0.5-comp-reg-stats-run_id.npy
# python -u -m premise.interval.run 6h comp_methods -acas 0.75 --run-id=run_id -ll 30 -t -cl 20 -ho 10 -st 0.005 -sp 3 --dump-stats out/stats/date/acas1-ref-stats-run_id.npy -v --epsilon 0.00001 -m out/models/date/acas1-comp-ref-run_id :: -acas 0.75 --run-id=run_id -ll 30 -t -cl 20 -ho 10 --dump-stats out/stats/date/acas1-comp-noref-stats-run_id.npy -v --epsilon 0.00001 -m out/models/date/acas1-comp-no-ref-run_id :: -acas 0.75 --run-id=run_id --l 20 --horizon 10 -m out/models/date/acas1-comp-reg-run_id --dump-stats out/stats/date/acas1-comp-reg-stats-run_id.npy

# low threshold
#python -u -m premise.interval.run 12h comp_methods -mc SnL-10x10 --run-id=run_id -t -st 0.001 -sp 3 --dump-stats out/stats/date/SnL-10x10-comp-ref-stats-run_id.npy -v -m out/models/date/SnL-10x10-comp-ref-run_id :: -mc SnL-10x10 --run-id=run_id -m out/models/date/SnL-10x10-comp-reg-run_id --dump-stats out/stats/date/SnL-10x10-comp-reg-stats-run_id.npy :: -mc SnL-10x10 --run-id=run_id -m out/models/date/SnL-10x10-comp-mle-run_id --dump-stats out/stats/date/SnL-10x10-comp-mle-stats-run_id.npy :: --mc SnL-10x10 --run-id=run_id --dump-model out/models/date/ --dump-stats out/stats/date/ --no-target 

