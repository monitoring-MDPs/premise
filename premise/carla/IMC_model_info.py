

#apptainer run ./my_container.sif 
#python3 /cephyr/users/skurka/Alvis/MODD/experiments/contextual_MODD/carla/lane_keeping/IMC_model_info.py


def get_states_and_transitions():
    colors = ['LLL', 'LLM', 'LLH', 'LML', 'LMM', 'LMH', 'LHL', 'LHM', 'LHH',
        'MLL', 'MLM', 'MLH', 'MML', 'MMM', 'MMH', 'MHL', 'MHM', 'MHH',
        'HLL', 'HLM', 'HLH', 'HML', 'HMM', 'HMH', 'HHL', 'HHM', 'HHH']

    speed = ['slowest','slow','fast','fastest']

    distance = ['d10', 'd20','d30','d40','d50','d60','d70']

    percived_distance = ['pd10', 'pd20','pd30','pd40','pd50','pd60','pd70']

    #collision = [True, False]

    all_states = []
    for x in colors: 
            for y in speed: 
                for z in distance: 
                    for v in percived_distance: 
                        #for u in collision: 
                            all_states.append(((x,y,z,v),v,False))


    all_states = [s for s in all_states if not (s[2] == 'd70' and s[1] == 'fastest')]
    all_states.append((('collision'),'collision',True))


    all_transitions = []

    for x in all_states[:-1]: 
            for y in all_states[:-1]: 
                if x[0][0] == y[0][0]: 
                    if  abs(int(x[0][2][1:]) - int(y[0][2][1:])) <=10 and (int(x[0][2][1:]) >= int(y[0][2][1:])):
                                    if x[0][1]== 'slowest' and (y[0][1]=='slowest' or y[0][1]=='slow'): 
                                        all_transitions.append((x,y))
                                    elif x[0][1]== 'slow' and y[0][1] != 'fastest':
                                        all_transitions.append((x,y)) 
                                    elif x[0][1]== 'fast' and y[0][1] != 'slowest':
                                        all_transitions.append((x,y)) 
                                    elif x[0][1]== 'fastest' and (y[0][1]=='fast' or y[0][1]=='fastest'):
                                        all_transitions.append((x,y))


    filtered_transitions = []

    for x in all_transitions: 
            if (x[0][0][3] == 'pd10' and x[0][0][1] == 'slowest') and x[1][0][1] != 'slowest': 
                continue  
            if (x[0][0][3] == 'pd10' and x[0][0][1] == 'slow') and not (x[1][0][1] == 'slowest' or x[1][0][1] == 'slow'): 
                continue
            if (x[0][0][3] == 'pd10' and x[0][0][1] == 'fast') and not (x[1][0][1] == 'fast' or x[1][0][1] == 'slow'): 
                continue
            if (x[0][0][3] == 'pd10' and x[0][0][1] == 'fastest') and not (x[1][0][1] == 'fast' or x[1][0][1] == 'fastest'): 
                continue
            filtered_transitions.append(x)

    all_transitions = filtered_transitions

    for x in all_states[:-1]: 
            all_transitions.append((x,all_states[-1]))
        
    all_transitions.append((all_states[-1], all_states[-1]))

    return all_states, all_transitions
       

if __name__ == '__main__':
      print(get_states_and_transitions())