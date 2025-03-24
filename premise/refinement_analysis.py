import numpy as np 
import matplotlib.pyplot as plt
import pickle

#refinement_results = np.load('/workspaces/premise/premise/examples/refinement_results.npy', allow_pickle=True)

with open('/workspaces/premise/premise/examples/refinement_results.pkl', 'rb') as f:
    refinement_results = pickle.load(f)

x = []
y1= []
y2 = []
y3 = []

for k in refinement_results.keys():
        x.append(k)
        y1.append(refinement_results[k][0])
        y2.append(refinement_results[k][1])
        y3.append(abs(refinement_results[k][3][1] - refinement_results[k][3][0]))

plt.plot(x, y1)
plt.xlabel("Refinement step")
plt.ylabel("Gap Values")

plt.title("Averge difference to samples (500)")

plt.show(block=True)


plt.plot(x, y2)
plt.xlabel("Refinement step")
plt.ylabel("Gap Values")

plt.title("Averge difference to risk on true model")


plt.show(block=True)

plt.plot(x, y3)
plt.xlabel("Refinement step")
plt.ylabel("Gap Values")

plt.title("Worst perfoming trace")

plt.show(block=True)     

for a in range(len(x)): 
    if all (abs(y1[a] - y1[a-z]) <0.02 for z in range(1,11)):
        print(a)

for a in range(len(x)): 
    if all (abs(y2[a] - y2[a-z]) <0.02 for z in range(1,11)):
        print(a)



plt.plot(x[15:], y1[15:])
plt.xlabel("Refinement step")
plt.ylabel("Gap Values")

plt.title("Averge difference to samples (500)")

plt.show(block=True)


plt.plot(x[15:], y2[15:])
plt.xlabel("Refinement step")
plt.ylabel("Gap Values")

plt.title("Averge difference to risk on true model")

plt.show(block=True)

