import stormpy as sp

from premise.interval.policy_iteration import policy_iter_imc
from premise.interval.interval import stormpy_ipomdp_to_imdp, stormpy_imdp_to_imc

model = sp.build_interval_model_from_drn("../premise/examples/brp-16-2.drn")

print(model)

imdp = stormpy_ipomdp_to_imdp(model)
imc = stormpy_imdp_to_imc(imdp)

env = sp.Environment()
int_env = sp.Environment()
int_env.solver_environment.minmax_solver_environment.method = (
    sp.MinMaxMethod.value_iteration
)

formulas = sp.parse_properties('Pmax=? [ F "target" ]')
res, pol = policy_iter_imc(imc, formulas[0], True, env)

task = sp.CheckTask(formulas[0].raw_formula, False)
res2 = sp.check_interval_mdp(imdp, task, int_env)

for s in range(len(imc.states)):
    diff = abs(res.at(s) - res2.at(s))
    if diff > 1e-6:
        print(s, res.at(s), res2.at(s), diff)
