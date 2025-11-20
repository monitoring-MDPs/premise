# python premise/demo.py \
#     --unfolding --exact --name "timing" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F \"crash\"]" --seed 0 --unfolding-mode storm-conditional --conditional-method bisection-advanced --promptness-deadline 100000 --verbose && \
# python premise/demo.py \
#     --unfolding --exact --name "timing" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --unfolding-mode storm-conditional --conditional-method bisection --promptness-deadline 100000 && \
# python premise/demo.py \
#     --unfolding --exact --name "timing" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --unfolding-mode storm-conditional --conditional-method restart --promptness-deadline 100000 && \
# python premise/demo.py \
#     --unfolding --exact --name "timing" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --unfolding-mode storm-conditional --conditional-method pi --promptness-deadline 100000 && \
# python premise/demo.py \
#     --unfolding --exact --name "timing" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --unfolding-mode rejection_sampling --promptness-deadline 100000

# python premise/demo.py --unfolding --exact --name "threshold" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --threshold 0.2 --unfolding-mode storm-conditional --conditional-method bisection --promptness-deadline 100000 && \
# python premise/demo.py \
#     --unfolding --exact --name "threshold" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --threshold 0.2 --unfolding-mode storm-conditional --conditional-method restart --promptness-deadline 100000 && \
# python premise/demo.py \
#     --unfolding --exact --name "threshold" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --threshold 0.2 --unfolding-mode storm-conditional --conditional-method pi --promptness-deadline 100000 && \
# python premise/demo.py \
#     --unfolding --exact --name "threshold" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --threshold 0.2 --unfolding-mode rejection_sampling --promptness-deadline 100000

## Profiling runs

# Non-exact
py-spy record --native -o profile_db_no_threshold.svg -- python premise/demo.py --unfolding --float --name "no-threshold" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --unfolding-mode storm-conditional --conditional-method bisection --mc-method value_iteration --promptness-deadline 100000
py-spy record --native -o profile_db.svg -- python premise/demo.py --unfolding --float --name "threshold" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --threshold 0.2 --unfolding-mode storm-conditional --conditional-method bisection --mc-method value_iteration --promptness-deadline 100000
py-spy record --native -o profile_db_restart.svg -- python premise/demo.py --unfolding --float --name "threshold" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --threshold 0.2 --unfolding-mode storm-conditional --conditional-method restart --promptness-deadline 100000

# Exact
py-spy record --native -o profile_no_threshold.svg -- python premise/demo.py --unfolding --exact --name "no-threshold" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --unfolding-mode storm-conditional --conditional-method bisection --mc-method value_iteration --promptness-deadline 100000
py-spy record --native -o profile.svg -- python premise/demo.py --unfolding --exact --name "threshold" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --threshold 0.2 --unfolding-mode storm-conditional --conditional-method bisection --mc-method value_iteration --promptness-deadline 100000
py-spy record --native -o profile_restart.svg -- python premise/demo.py --unfolding --exact --name "threshold" --model premise/examples/airportA-3.nm --constants "DMAX=5,PMAX=5" --risk "Pmax=? [F<=30 \"crash\"]" --seed 0 --threshold 0.2 --unfolding-mode storm-conditional --conditional-method restart --promptness-deadline 100000
