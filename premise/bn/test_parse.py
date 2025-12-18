import benchmark

test_str = """
Storm 1.11.1 (dev)

Date: Tue Dec 16 13:49:17 2025
Command line arguments: -drn ./premise/examples/transformed-mdp/crowds_3-5.drn --prop 'Pmax=? [F "observe0Greater1"];Pmax=? [F "observeIGreater1" || F "observe0Greater1"]' --conditional pi '--minmax:precision' 1e-6 --dot-maxwidth 0
Current working directory: /home/luko/Documents/Cond-Prob-Project/premise

Time for model construction: 0.164s.

-------------------------------------------------------------- 
Model type:     MDP (sparse)
States:         1772
Transitions:    6602
Choices:        3242
Reward Models:  none
State Labels:   5 labels
   * deadlock -> 56 item(s)
   * observeIGreater1 -> 806 item(s)
   * observeOnlyTrueSender -> 337 item(s)
   * observe0Greater1 -> 425 item(s)
   * init -> 1 item(s)
Choice Labels:  none
-------------------------------------------------------------- 

Model checking property "1": Pmax=? [F "observe0Greater1"] ...
Result (for initial states): 0.9999993807
Time for model checking: 0.002s.

Model checking property "2": Pmax=? [(F "observeIGreater1") || (F "observe0Greater1")] ...
WARN  (ConditionalHelper.cpp:1132): Potential numerical issues: the probability to reach the target is greater than the probability to reach the condition. Difference is 1.10617e-07.
WARN  (ConditionalHelper.cpp:1132): Potential numerical issues: the probability to reach the target is greater than the probability to reach the condition. Difference is 4.45461e-10.
WARN  (ConditionalHelper.cpp:1132): Potential numerical issues: the probability to reach the target is greater than the probability to reach the condition. Difference is 2.57503e-10.
WARN  (ConditionalHelper.cpp:1132): Potential numerical issues: the probability to reach the target is greater than the probability to reach the condition. Difference is 1.04347e-10.
WARN  (ConditionalHelper.cpp:1132): Potential numerical issues: the probability to reach the target is greater than the probability to reach the condition. Difference is 8.11849e-12.
WARN  (ConditionalHelper.cpp:1132): Potential numerical issues: the probability to reach the target is greater than the probability to reach the condition. Difference is 1.32562e-11.
WARN  (ConditionalHelper.cpp:1132): Potential numerical issues: the probability to reach the target is greater than the probability to reach the condition. Difference is 1.72719e-10.
WARN  (ConditionalHelper.cpp:1132): Potential numerical issues: the probability to reach the target is greater than the probability to reach the condition. Difference is 3.63328e-11.
Result (for initial states): 1.000000001
Time for model checking: 0.007s."""

print(benchmark._parse_storm_output(test_str, [("", "", "", "", "", "")], 0))
