import pickle
import os
from IMC_model_info import get_states_and_transitions


with open("/workspaces/premise/premise/carla/carla_samples/all_data.pkl", "rb") as g:
    all_data = pickle.load(g)

#output_dir = '/workspaces/premise/premise/carla/carla_samples' 
#output_file = os.path.join(output_dir, 'all_data.pkl')
#os.makedirs(output_dir, exist_ok=True)

#with open(output_file, "wb") as f:
#    pickle.dump(all_data, f)


all_states = get_states_and_transitions()[0]

frequency= {}
for s in all_states: 
    frequency[s] = 0


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

prefix_freq = {}

for x in all_data: 
    if x[0][0][2] == "HHH": 
        prefix_freq[(x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1],x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1],x[20][1],x[21][1],x[22][1],x[23][1],x[24][1],x[25][1],x[26][1],x[27][1],x[28][1],x[29][1],x[30][1],x[31][1],x[32][1],x[33][1],x[34][1],x[35][1],x[36][1],x[37][1],x[38][1],x[39][1], x[40][1],x[41][1],x[42][1],x[43][1],x[44][1],x[45][1],x[46][1],x[47][1],x[48][1],x[49][1])] = 0 


for k in prefix_freq.keys():
    for x in all_data: 
        if x[0][0][2] == "HHH": 
            if (x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1],x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1],x[20][1],x[21][1],x[22][1],x[23][1],x[24][1],x[25][1],x[26][1],x[27][1],x[28][1],x[29][1],x[30][1],x[31][1],x[32][1],x[33][1],x[34][1],x[35][1],x[36][1],x[37][1],x[38][1],x[39][1], x[40][1],x[41][1],x[42][1],x[43][1],x[44][1],x[45][1],x[46][1],x[47][1],x[48][1],x[49][1]) == k: 
                prefix_freq[k] +=1 


for x in prefix_freq.keys(): 
    if prefix_freq[x] > 1:
        print(x, prefix_freq[x])

for x in all_data: 
    if (x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1],x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1],x[20][1],x[21][1],x[22][1],x[23][1],x[24][1],x[25][1],x[26][1],x[27][1],x[28][1],x[29][1],x[30][1],x[31][1],x[32][1],x[33][1],x[34][1],x[35][1],x[36][1],x[37][1],x[38][1],x[39][1], x[40][1],x[41][1],x[42][1],x[43][1],x[44][1],x[45][1],x[46][1],x[47][1],x[48][1],x[49][1]) == (('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50')):
        print(x[-100])


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

