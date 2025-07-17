from pathlib import Path
import pickle
import numpy as np

def get_transition_count(stats_path, model_def):
    transition_stats = []
    path = Path(stats_path) 
    if path.is_dir():
        paths = path.iterdir()
    else:
        paths = [path]

    for stat_path in paths:
        if model_def in str(stat_path):
            if '-ref-' in str(stat_path) or '-refsplit-' in str(stat_path):
                    try:
                        data = np.load(stat_path, allow_pickle=True).item()
                    except AttributeError:
                        data = pickle.load(stat_path.open("rb"))
                    try:
                        transition_stats.append(np.sum(data["transitions_learned"]))
                    except Exception as e:
                        print(f"Error processing {stat_path}: {e} ({data})")

    return max(transition_stats)
                
if __name__ == "__main__":
    transition_count = get_transition_count('/workspaces/premise/out/stats/2025-07-08_08-55-26', 'SnL-10x10')
    print(transition_count)