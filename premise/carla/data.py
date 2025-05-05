import pickle
from IMC_model_info import get_states_and_transitions

all_data = [] 


with open("/workspaces/premise/premise/carla/carla_samples/200.pkl", "rb") as f:
    data1 = pickle.load(f)
    for x in data1: 
        all_data.append(x)

with open("/workspaces/premise/premise/carla/carla_samples/400.pkl", "rb") as g:
    data2 = pickle.load(g)
    for y in data2: 
        all_data.append(y)


with open("/workspaces/premise/premise/carla/carla_samples/600.pkl", "rb") as h:
    data3 = pickle.load(h)
    for z in data3: 
        all_data.append(z)

with open("/workspaces/premise/premise/carla/carla_samples/800.pkl", "rb") as y:
    data4 = pickle.load(y)
    for a in data4: 
        all_data.append(a)


with open("/workspaces/premise/premise/carla/carla_samples/1000.pkl", "rb") as g:
    data5 = pickle.load(g)
    for b in data5: 
        all_data.append(b)

all_states = get_states_and_transitions()[0]


frequency= {}
for s in all_states: 
    frequency[s] = 0

for k in frequency.keys():
    for t in all_data: 
        for x in t: 
            if x==k: 
                frequency[k]+=1


colors = ['LLL', 'LLM', 'LLH', 'LML', 'LMM', 'LMH', 'LHL', 'LHM', 'LHH',
        'MLL', 'MLM', 'MLH', 'MML', 'MMM', 'MMH', 'MHL', 'MHM', 'MHH',
        'HLL', 'HLM', 'HLH', 'HML', 'HMM', 'HMH', 'HHL', 'HHM', 'HHH']

color_freq = {}
for c in colors: 
   color_freq[c] = 0 

for x in all_data:
    for c in colors:
        if x[0][0][2] == c: 
            color_freq[c] +=1 

print(color_freq)


#ood = 0 
#ood_diff = 0  

#for t in all_data: 
#    for s in t: 
#        if s!= (('collision'),'collision',True):
#            if s[0][2] == colors[26]:
#                    ood += 1
#                    ood_diff += abs(int(s[0][1][1:]) - int(s[0][3][2:])) 
        
            
#print(ood)
#print(ood_diff/ood)


#LLL - 14.448695814474359   #>100
#LLM - 3.39
#LLH - 2.12
#LML - 2.12
#LMM - 6.865377730039384
#LMH - 18.89952153110048
#LHL - none 
#LHM - 29.033989266547405
#LHH - none 

#MLL - 13.76736111111111
#MLM - 3.31 
#MLH - none
#MML - 9.795008912655971
#MMM - 11.585586794605113    #>100
#MMH - 4.560862865947612 
#MHL - 3.04
#MHM - 8.29780146568954
#MHH - 4.488188976377953

#HLL - 12.542934618861104
#HLM - none
#HLH - none
#HML - 12.6298044504383
#HMM - 12.08073115003808
#HMH - 4.705882352941177
#HHL - 4.488836662749706
#HHM - 8.155105973025048
#HHH - 11.419307087724246   #>100

