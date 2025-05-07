import pickle 
import os

all_data = []

with open("/workspaces/premise/premise/carla/carla_samples/batch_10.pkl", "rb") as g:
    data = pickle.load(g)
    for s in data: 
        if len(s) == 250:
            all_data.append(s) 

with open("/workspaces/premise/premise/carla/carla_samples/batch_11.pkl", "rb") as g:
    data = pickle.load(g)
    for s in data: 
        if len(s) == 250:
            all_data.append(s) 


print(len(all_data))

#output_dir = '/workspaces/premise/premise/carla/carla_samples' 
#output_file = os.path.join(output_dir, 'test_data.pkl')
#os.makedirs(output_dir, exist_ok=True)

#with open(output_file, "wb") as f:
#    pickle.dump(all_data, f)




#with open("/workspaces/premise/premise/carla/carla_samples/HHH_0_50.pkl", "rb") as g:
#    data = pickle.load(g)
#    for s in data: 
#        if len(s) == 250:
#            all_data.append(s)

#with open("/workspaces/premise/premise/carla/carla_samples/HHH_50_100.pkl", "rb") as g:
#    data = pickle.load(g)
#    for s in data: 
#        if len(s) == 250:
#            all_data.append(s)

#with open("/workspaces/premise/premise/carla/carla_samples/HHH_100_150.pkl", "rb") as g:
#    data = pickle.load(g)
#    for s in data: 
#        if len(s) == 250:
#            all_data.append(s) 

#with open("/workspaces/premise/premise/carla/carla_samples/HHH_150_200.pkl", "rb") as g:
#    data = pickle.load(g)
#   for s in data: 
#        if len(s) == 250:
#            all_data.append(s)

#with open("/workspaces/premise/premise/carla/carla_samples/HHH_200_250.pkl", "rb") as g:
#    data = pickle.load(g)
#    for s in data: 
#        if len(s) == 250:
#            all_data.append(s)



#freq = {}

#for x in all_data: 
    #freq[(x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1], x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1])] = 0 
    #freq[(x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1], x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1],x[20][1],x[21][1],x[22][1],x[23][1],x[24][1])] = 0 
    #freq[(x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1], x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1],x[20][1],x[21][1],x[22][1],x[23][1],x[24][1],x[25][1],x[26][1],x[27][1],x[28][1],x[29][1])] = 0
    #freq[(x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1], x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1],x[20][1],x[21][1],x[22][1],x[23][1],x[24][1],x[25][1],x[26][1],x[27][1],x[28][1],x[29][1],x[30][1],x[31][1],x[32][1],x[33][1],x[34][1],x[35][1],x[36][1],x[37][1],x[38][1],x[39][1])] = 0 
    #freq[(x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1], x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1],x[20][1],x[21][1],x[22][1],x[23][1],x[24][1],x[25][1],x[26][1],x[27][1],x[28][1],x[29][1],x[30][1],x[31][1],x[32][1],x[33][1],x[34][1],x[35][1],x[36][1],x[37][1],x[38][1],x[39][1],x[40][1],x[41][1],x[42][1],x[43][1],x[44][1],x[45][1],x[46][1],x[47][1],x[48][1],x[49][1])] = 0 


#for k in freq.keys(): 
    #for x in all_data: 
        #if (x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1], x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1]) == k: 
        #if (x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1], x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1],x[20][1],x[21][1],x[22][1],x[23][1],x[24][1]) == k: 
        #if (x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1], x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1],x[20][1],x[21][1],x[22][1],x[23][1],x[24][1],x[25][1],x[26][1],x[27][1],x[28][1],x[29][1]) == k: 
        #if (x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1], x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1],x[20][1],x[21][1],x[22][1],x[23][1],x[24][1],x[25][1],x[26][1],x[27][1],x[28][1],x[29][1],x[30][1],x[31][1],x[32][1],x[33][1],x[34][1],x[35][1],x[36][1],x[37][1],x[38][1],x[39][1]) == k: 
        #if (x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1], x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1],x[20][1],x[21][1],x[22][1],x[23][1],x[24][1],x[25][1],x[26][1],x[27][1],x[28][1],x[29][1],x[30][1],x[31][1],x[32][1],x[33][1],x[34][1],x[35][1],x[36][1],x[37][1],x[38][1],x[39][1],x[40][1],x[41][1],x[42][1],x[43][1],x[44][1],x[45][1],x[46][1],x[47][1],x[48][1],x[49][1]) == k:
        #   freq[k] += 1

#for k in freq: 
#    if freq[k] > 20: 
#        print(k, freq[k])



#for x in all_data: 
#    if (x[0][1],x[1][1],x[2][1],x[3][1],x[4][1],x[5][1],x[6][1],x[7][1],x[8][1],x[9][1],x[10][1],x[11][1],x[12][1], x[13][1],x[14][1],x[15][1],x[16][1],x[17][1],x[18][1],x[19][1],x[20][1],x[21][1],x[22][1],x[23][1],x[24][1],x[25][1],x[26][1],x[27][1],x[28][1],x[29][1]) == (('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60')):
#        if x[-150] == ('collision', 'collision', True): 
#            print('collsion')


#(('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'))
#len = 20 
#horizon = 80
#6/120 = 0.05
#horizon = 70 
#0/120

#(('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'), ('HHH', 'pd50'))
#len = 20 
#horizon = 80
#4/60 = 0.066
#horizon = 50
#0/60

#(('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'))
#len = 20
#horizon = 80
#0/43 = 0.0

#(('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'), ('HHH', 'pd60'))
#len = 30 
#horizon = 70
#2/76 = 0.026



#(('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'), ('HHH', 'pd10'))
#len = 50 
#horizon = 25
#0/40
#horizon = 50
#0/40




