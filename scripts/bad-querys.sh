
# # 1259 states
# ../storm-cond/build/bin/storm -drn premise/examples/BN-benchmarks-mdp/alarm.drn --prop 'Pmax=? [F "HREKG0" || F "CVP0"]' --conditional bisection --exact
# ../storm-cond/build/bin/storm -drn premise/examples/BN-benchmarks-mdp/alarm.drn --prop 'Pmax=? [F "HREKG0" || F "CVP0"]' --conditional restart --exact

# # 37569 states
# ../storm-cond/build/bin/storm -drn premise/examples/BN-benchmarks-mdp/water.drn --prop 'Pmax=? [F "CBODD_12_452" || F "CNOD_12_451"]' --conditional restart --exact
# ../storm-cond/build/bin/storm -drn premise/examples/BN-benchmarks-mdp/water.drn --prop 'Pmax=? [F "CBODD_12_452" || F "CNOD_12_451"]' --conditional conditional --exact

# # 1450 states
# ../storm-cond/build/bin/storm -drn premise/examples/BN-benchmarks-mdp/child.drn --prop 'Pmax=? [F "RUQO20" || F "XrayReport3"]' --conditional restart --exact
# ../storm-cond/build/bin/storm -drn premise/examples/BN-benchmarks-mdp/child.drn --prop 'Pmax=? [F "RUQO20" || F "XrayReport3"]' --conditional bisection --exact

# 130 states
# query="Pmax=? [F \"Akt0\" || F \"P382\"]"
# ../storm-cond/build/bin/storm -drn premise/examples/BN-benchmarks-mdp/sachs.drn --prop "$query" --conditional bisection --exact --exportresult sachs-bisection.json
# ../storm-cond/build/bin/storm -drn premise/examples/BN-benchmarks-mdp/sachs.drn --prop "$query" --conditional restart --exact --exportresult sachs-restart.json

# 22 states
query='Pmax=? [F "A0" || F "T0"]'
../storm-cond/build/bin/storm -drn premise/examples/BN-benchmarks-mdp/survey.drn --prop "$query" --conditional bisection --exact --exportresult survey-bisection.json --exportscheduler s-bisection.txt
../storm-cond/build/bin/storm -drn premise/examples/BN-benchmarks-mdp/survey.drn --prop "$query" --conditional bisection-advanced --exact --exportresult survey-bisection-advanced.json --exportscheduler s-bisection-advanced.txt
../storm-cond/build/bin/storm -drn premise/examples/BN-benchmarks-mdp/survey.drn --prop "$query" --conditional restart --exact --exportresult survey-restart.json --exportscheduler s-restart.txt

echo "\nBisection:"
jq '.[0].v' survey-bisection.json
echo "Bisection-Advanced:"
jq '.[0].v' survey-bisection-advanced.json
echo "Restart:"
jq '.[0].v' survey-restart.json
