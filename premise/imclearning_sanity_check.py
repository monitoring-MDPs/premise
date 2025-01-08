from curses import nl
import scenic
import tempfile
import pathlib

from scenic.simulators.newtonian import NewtonianSimulator

scenario = scenic.scenarioFromFile('Scenic/examples/driving/car_CTE.scenic',
                                   model='scenic.simulators.newtonian.driving_model',
                                   mode2D=True)

samples = []

for x in range(1000): 

    scene, _ = scenario.generate()
    simulator = NewtonianSimulator()
    simulation = simulator.simulate(scene, maxSteps=155)
    if simulation:  # `simulate` can return None if simulation fails
        result = simulation.result
        
        trace = [] 
        for n in range(50,150):
            state = []
            
            if result.records['CTE'][n][1] < 0.1: 
                state.append('0.0-0.1')
            if 0.1 <= result.records['CTE'][n][1] < 0.2: 
                state.append('0.1-0.2')
            if 0.2 <= result.records['CTE'][n][1] < 0.3: 
                state.append('0.2-0.3')
            if 0.3 <= result.records['CTE'][n][1] < 0.4: 
                state.append('0.3-0.4')
            if 0.4 <= result.records['CTE'][n][1] < 0.5: 
                state.append('0.4-0.5')  
            if 0.5 <= result.records['CTE'][n][1] < 0.6: 
                state.append('0.5-0.6')
            if 0.6 <= result.records['CTE'][n][1] < 0.7: 
                state.append('0.6-0.7')
            if 0.7 <= result.records['CTE'][n][1] < 0.8: 
                state.append('0.7-0.8') 
            if 0.8 <= result.records['CTE'][n][1] < 0.9: 
                state.append('0.8-0.9')
            if 0.9 <= result.records['CTE'][n][1] < 1.0: 
                state.append('0.7-0.8')
            if 1.0 <= result.records['CTE'][n][1] < 1.1: 
                state.append('1.0-1.1')
            if 1.1 <= result.records['CTE'][n][1] < 1.2: 
                state.append('1.1-1.2') 
            if 1.2 <= result.records['CTE'][n][1] < 1.3: 
                state.append('1.2-1.3')
            if 1.3 <= result.records['CTE'][n][1] < 1.4: 
                state.append('1.3-1.4')
            if 1.4 <= result.records['CTE'][n][1] < 1.5: 
                state.append('1.4-1.5')
            if 1.5 <= result.records['CTE'][n][1] < 1.6: 
                state.append('1.5-1.6')
            if 1.6 <= result.records['CTE'][n][1] < 1.7: 
                state.append('1.6-1.7')
            if 1.7 <= result.records['CTE'][n][1] < 1.8: 
                state.append('1.6-1.8')
            if 1.8 <= result.records['CTE'][n][1] < 1.9: 
                state.append('1.8-1.9')
            if 1.9 <= result.records['CTE'][n][1] < 2.0: 
                state.append('1.9-2.0')
            if result.records['CTE'][n+3][1] >= 0.8 or result.records['CTE'][n+2][1] >= 0.8 or result.records['CTE'][n+1][1] >= 0.8: 
                state.append('invasion_within_3')
            if result.records['CTE'][n+3][1] < 0.8 and result.records['CTE'][n+2][1] < 0.8 and result.records['CTE'][n+1][1] < 0.8: 
                state.append('no_invasion_within_3')
                
            
                
            trace.append(state) 
    samples.append(trace) 

all_states = []

for x in (('0.0-0.1'),('0.1-0.2'),('0.2-0.3'),('0.3-0.4'),('0.4-0.5'),('0.5-0.6'),('0.6-0.7'),('0.7-0.8'),('0.8-0.9'),('0.9-1.0'),('1.0-1.1'),('1.1-1.2'),('1.2-1.3'),('1.3-1.4'),('1.4-1.5'),('1.5-1.6'),('1.6-1.7'),('1.7-1.8'),('1.8-1.9'),('1.9-2.0')):
    for y in (('invasion_within_3'), ('no_invasion_within_3')):
        all_states.append([])
        all_states[-1].append(x) 
        all_states[-1].append(y)

epsilon = 1/1000

i_nl = 10   #initial lower bound of strength interval 
i_nu = 20   #initial upper bound of strength interval 

i_i_nl = 5  #initial lower bound of strength interval for initial distribution
i_i_nu = 10

initial_count = {}

for s in all_states:
    initial_count[tuple(s)] = 0


for t in samples:
    for s in all_states: 
        if s == t[0]: 
            initial_count[tuple(s)] +=1

initial_nl = {}
initial_nu = {}

for s in all_states: 
    initial_nl[tuple(s)] = i_i_nl
    initial_nu[tuple(s)] = i_i_nu
    

initial_interval = {}

for s in all_states:
    initial_interval[tuple(s)] = [epsilon, 1-epsilon]

for n in initial_count.keys():
    for i in initial_interval.keys():
        if n == i:
            if any((initial_count[x]/1000) < initial_interval[n][0] for x in  initial_count.keys()): 
                initial_interval[n][0] = (((initial_interval[n][0] * initial_nl[n]) + initial_count[n])/(initial_nl[n] + 1000))
            else: 
                initial_interval[n][0] = (((initial_interval[n][0] * initial_nu[n]) + initial_count[n])/(initial_nu[n] + 1000))  

for n in initial_count.keys():
    for i in initial_interval.keys():
        if n == i:
            if any((initial_count[x]/1000) > initial_interval[n][1] for x in initial_count.keys()): 
                initial_interval[n][0] = (((initial_interval[n][0] * initial_nl[n]) + initial_count[n])/(initial_nl[n] + 1000))
            else: 
                initial_interval[n][1] = (((initial_interval[n][1] * initial_nu[n]) + initial_count[n])/(initial_nu[n] + 1000))

for s in all_states: 
    initial_nl[tuple(s)] += 1000
    initial_nu[tuple(s)] += 1000

transition_count = {} 

for s in all_states: 
    transition_count[tuple(s)] = 0

for t in samples: 
    for s in t[0:100]:
         for k in transition_count.keys():
             if tuple(s) == k: 
                 transition_count[tuple(s)] +=1

tau_count = {} 

for a in transition_count.keys():
    for b in transition_count.keys():
        tau_count[a,b] = 0 


for t in samples: 
   for n in range(len(t) -1):
       for a in transition_count.keys():
           for b in transition_count.keys():
               if t[n] == list(a) and t[n+1] == list(b):
                   tau_count[a,b] += 1
        

interval = {}

for k in tau_count.keys():
    interval[k] = [epsilon, 1-epsilon]

nl = {} 
nu = {} 

for k in tau_count.keys(): 
    nl[k] = i_nl 
    nu[k] = i_nu 
    
for n in transition_count.keys():
    for m in tau_count.keys():
        for i in interval.keys():
            if n == m[0] and m == i:
                if transition_count[n] != 0: 
                    if any((tau_count[x]/transition_count[n] < interval[x][0] and x[0] == n) for x in tau_count.keys()):
                        interval[i][0] = ((nl[m] * interval[i][0]) + tau_count[m])/(nl[m] + transition_count[n])
                    else: 
                        interval[i][0] = ((nu[m] * interval[i][0]) + tau_count[m])/(nu[m] + transition_count[n])
                    

for n in transition_count.keys():
    for m in tau_count.keys():
        for i in interval.keys():
            if n == m[0] and m == i:
                if transition_count[n] != 0: 
                    if any((tau_count[x]/transition_count[n] > interval[x][1] and x[0] == n) for x in tau_count.keys()):
                        interval[i][1] = ((nl[m] * interval[i][1]) + tau_count[m])/(nl[m] + transition_count[n])
                    else: 
                        interval[i][1] = ((nu[m] * interval[i][1]) + tau_count[m])/(nu[m] + transition_count[n])


for k in transition_count.keys(): 
    for t in tau_count.keys(): 
        if t[0] == k:
            nl[t] += transition_count[k]
            nu[t] += transition_count[k]

