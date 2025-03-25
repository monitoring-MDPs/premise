import matplotlib.pyplot as plt
import pickle

with open('/workspaces/premise/premise/examples/refinement_per_trace_results.pkl', 'rb') as f:
    refinement_results = pickle.load(f)

with open('/workspaces/premise/premise/examples/refinement_per_trace_sample_results.pkl', 'rb') as f:
    refinement_results_samples = pickle.load(f)

with open('/workspaces/premise/premise/examples/refinement_per_trace_new_sample_index.pkl', 'rb') as f:
    refinement_per_trace_new_sample_index = pickle.load(f)


y1 = refinement_results
y2 = refinement_results_samples


difference = []
for a in range(len(y1)): 
    difference.append(abs(y1[a] - y2[a]))

y3 = refinement_per_trace_new_sample_index

x = list(range(1,len(y1)+1))

plt.plot(x, difference)
for x in y3:
    plt.axvline(x, color='r')
plt.xlabel("Refinement step")
plt.ylabel("Gap Values")

plt.title("Averge difference to samples (1000)")

plt.show(block=True)

print(y3)
