import csv

class TraceMapper:
    """
    Class that helps to translate between low-level observations and high-level observations.
    """
    def __init__(self, model):
        self._model = model
        self._inverse_observation_valuations = None # Is constructed below.
        self._obsval = self._model.observation_valuations # Caching for efficiency (due to some stormpy issue)
        self._construct_inverse_observation_valuations()

    def get_observation_names(self):
        """
        Gives the names of the high-level observations
        """
        obs_tuple = model.observation_valuations.get_string(0, pretty=True)[1:-1].replace("\t", " ").split(" & ")
        obs_tuple_pruned = [obs.split("=")[0] if "=" in obs else obs if "!" not in obs else obs[1:] for obs
                            in obs_tuple]
        return obs_tuple_pruned

    def _construct_inverse_observation_valuations(self):
        """
        Constructs an explicit inverse observation valuation and stores this for simple access.

        TODO: This implementation is not particularly efficient.
        """
        self._inverse_observation_valuations = {}
        for o in range(self._obsval.get_nr_of_states()):
            self._inverse_observation_valuations[self._lowlevel_to_highlevel(o)] = o

    def _lowlevel_to_highlevel(self, observation:int):
        obs_tuple = self._obsval.get_string(observation, pretty=True)[1:-1].replace("\t", " ").split(" & ")
        return tuple(int(obs.split("=")[1]) if "=" in obs else True if "!" not in obs else False for obs in
                            obs_tuple)

    def annotated_trace_to_highlevel(self, trace):
        high_level_trace = []
        for observation, risk in trace:
            high_level_trace.append((self._lowlevel_to_highlevel(observation), risk))
        return high_level_trace

    def trace_to_high_level(self, trace):
        """
        Takes a trace and converts it to a high level trace.
        """
        high_level_trace = []
        for observation in trace:
            high_level_trace.append(self._lowlevel_to_highlevel(observation))
        return high_level_trace

    def trace_from_high_level_trace(self, high_level_trace):
        """
        Takes a high level trace and converts it to a low level trace.
        """
        trace = []
        for hl_obs in high_level_trace:
            low_obs = self._inverse_observation_valuations.get(hl_obs)
            trace.append(low_obs)
        return trace


def export_annotated_high_level(annotated_trace, model, trace_file):
    with open(trace_file, 'w', newline='') as csvfile:
        csvfilewriter = csv.writer(csvfile, delimiter=',',
                                   quotechar='|', quoting=csv.QUOTE_MINIMAL)
        obs_tuple = model.observation_valuations.get_string(0, pretty=True)[1:-1].replace("\t", " ").split(" & ")
        obs_tuple_pruned = [obs.split("=")[0] if "=" in obs else obs if "!" not in obs else obs[1:] for obs
                            in obs_tuple]
        csvfilewriter.writerow(obs_tuple_pruned + ["risk"])
        for (obs, risk) in annotated_trace:
            csvfilewriter.writerow(list(obs) + [risk])
