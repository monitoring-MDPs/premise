import csv


def _observation_variables(observation_valuations):
    return observation_valuations.get_all_variables()


def observation_names(model):
    """
    Gives the names of the high-level observations (i.e., the observable variables)
    """
    return [var.name for var in _observation_variables(model.observation_valuations)]


class TraceMapper:
    """
    Class that helps to translate between low-level observations and high-level observations.
    """
    def __init__(self, model):
        self._model = model
        self._inverse_observation_valuations = None # Is constructed below.
        self._obsval = self._model.observation_valuations # Caching for efficiency (due to some stormpy issue)
        self._obsvars = _observation_variables(self._obsval)
        self._construct_inverse_observation_valuations()

    def get_observation_names(self):
        """
        Gives the names of the high-level observations
        """
        return observation_names(self._model)

    def _construct_inverse_observation_valuations(self):
        """
        Constructs an explicit inverse observation valuation and stores this for simple access.

        TODO: This implementation is not particularly efficient.
        """
        self._inverse_observation_valuations = {}
        for o in range(self._obsval.get_number_of_entities()):
            self._inverse_observation_valuations[self._lowlevel_to_highlevel(o)] = o

    def _lowlevel_to_highlevel(self, observation:int):
        return tuple(self._obsval.get_value(observation, var) for var in self._obsvars)

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
        csvfilewriter.writerow(observation_names(model) + ["risk"])
        for (obs, risk) in annotated_trace:
            csvfilewriter.writerow(list(obs) + [risk])
