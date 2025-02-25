epsilon = 1/1000 

i_i_nl = 5  #initial lower bound of strength interval for initial distribution
i_i_nu = 10 #initial upper bound of strength interval for initial distribution

i_nl = 10   #initial lower bound of strength interval 
i_nu = 20   #initial upper bound of strength interval 

#all_states = []
#samples = []


def premilinaries(epsilon, i_i_nl, i_i_nu, i_nl, i_nu, all_states): 

    initial_interval = {}

    for s in all_states:
        initial_interval[tuple(s)] = [epsilon, 1-epsilon]

    strenght_interval_initial = {}

    for s in all_states: 
        strenght_interval_initial[tuple(s)] = [i_i_nl, i_i_nu] 

    interval = {}

    for a in all_states:
        for b in all_states: 
            interval[tuple(a), tuple(b)] = [epsilon, 1-epsilon]

    strenght_interval = {} 

    for a in all_states:
        for b in all_states: 
            strenght_interval[tuple(a), tuple(b)] = [i_nl, i_nu]

    return initial_interval, strenght_interval_initial, interval, strenght_interval
    

def initial_interval_learning(all_states, samples, strenght_interval_initial, initial_interval): 

    trace_num = len(samples) #number of traces in a sample

    initial_count = {}  

    for s in all_states:
        initial_count[tuple(s)] = 0  #DOES IT NEED TO BE A TUPLE? 

    for t in samples:
        for s in all_states: 
            if s == t[0]: 
                initial_count[tuple(s)] +=1  
    
    for n in initial_count.keys():  #learns the lower bound of the initial interval 
        for i in initial_interval.keys():
            if n == i:
                if any((initial_count[x]/trace_num) < initial_interval[n][0] for x in  initial_count.keys()): 
                    initial_interval[n][0] = (((initial_interval[n][0] * strenght_interval_initial[n][0]) + initial_count[n])/(strenght_interval_initial[n][0] + trace_num))
                else: 
                    initial_interval[n][0] = (((initial_interval[n][0] * strenght_interval_initial[n][1]) + initial_count[n])/(strenght_interval_initial[n][1] + trace_num))  

    for n in initial_count.keys():  #learns the upper bound of the initial interval 
        for i in initial_interval.keys():
            if n == i:
                if any((initial_count[x]/trace_num) > initial_interval[n][1] for x in initial_count.keys()): 
                    initial_interval[n][1] = (((initial_interval[n][1] * strenght_interval_initial[n][0]) + initial_count[n])/(strenght_interval_initial[n][0] + trace_num))
                else: 
                    initial_interval[n][1] = (((initial_interval[n][1] * strenght_interval_initial[n][1]) + initial_count[n])/(strenght_interval_initial[n][1] + trace_num))


    for s in all_states:  #updates strength intervals 
        strenght_interval_initial[tuple(s)][0] += trace_num
        strenght_interval_initial[tuple(s)][1] += trace_num

    return initial_interval, strenght_interval_initial
    
def interval_learning(all_states, samples, interval, strenght_interval): 

    trace_num = len(samples) #number of traces in a sample
    trace_len = len(samples[0])

    transition_count = {} 

    for s in all_states: 
        transition_count[tuple(s)] = 0

    for t in samples: 
        for s in t[:-1]:
            for k in transition_count.keys():
                if tuple(s) == k:
                    transition_count[tuple(s)] +=1

    tau_count = {} 

    for a in transition_count.keys():
        for b in transition_count.keys():
            tau_count[a,b] = 0 

    for a in transition_count.keys():
        for b in transition_count.keys():
            for x in range(trace_num): 
                for y in range(trace_len-1):
                    if samples[x][y] == list(a) and samples[x][y+1] == list(b):
                        tau_count[a,b] += 1

    
    for n in transition_count.keys(): #learns the lower bound of the interval 
        for m in tau_count.keys():
            for i in interval.keys():
                if n == m[0] and m == i:
                    if transition_count[n] != 0: 
                        if any((tau_count[x]/transition_count[n] < interval[x][0] and x[0] == n) for x in tau_count.keys()):
                            interval[i][0] = ((strenght_interval[m][0] * interval[i][0]) + tau_count[m])/(strenght_interval[m][0] + transition_count[n])  #FIX THE USE OF nl and nu
                        else: 
                            interval[i][0] = ((strenght_interval[m][1] * interval[i][0]) + tau_count[m])/(strenght_interval[m][1] + transition_count[n])  #FIX THE USE OF nl and nu
                    
    for n in transition_count.keys():#learns the upper bound of the interval 
        for m in tau_count.keys():
            for i in interval.keys():
                if n == m[0] and m == i:
                    if transition_count[n] != 0: 
                        if any((tau_count[x]/transition_count[n] > interval[x][1] and x[0] == n) for x in tau_count.keys()):
                            interval[i][1] = ((strenght_interval[m][0] * interval[i][1]) + tau_count[m])/(strenght_interval[m][0] + transition_count[n])  
                        else: 
                            interval[i][1] = ((strenght_interval[m][1] * interval[i][1]) + tau_count[m])/(strenght_interval[m][1] + transition_count[n])  

    for k in transition_count.keys(): #updates strength intervals 
        for t in tau_count.keys(): 
            if t[0] == k:
                strenght_interval[t][0] += transition_count[k]
                strenght_interval[t][1] += transition_count[k]

    return interval, strenght_interval


premilinaries(epsilon, i_i_nl, i_i_nu, i_nl, i_nu, all_states)
initial_interval, strenght_interval_initial, interval, strenght_interval = premilinaries(epsilon, i_i_nl, i_i_nu, i_nl, i_nu, all_states)

initial_interval_learning(all_states, samples, strenght_interval_initial, initial_interval)
interval_learning(all_states, samples, interval, strenght_interval)

