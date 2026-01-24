import typing

from premise.monitor import MonitorTimeOutException


class Oracle:
    """
    This class provides an interface for running a premise-based oracle as a backend for an active learner.
    """
    def __init__(self, monitor):
        self._monitor = monitor

    def membership(self, trace : typing.List[int], threshold : float = None, intermediate_results : bool =False):
        """
        Provides the quantitative response after the trace.
        """
        result = [self._monitor.initialize(trace[0])]
        for t in trace[1:-1]:
            result.append(self._monitor.step(t, compute_risk=intermediate_results))
        try:
            result.append(self._monitor.step(trace[-1], compute_risk=True))
        except MonitorTimeOutException:
            result.append(None)
        if threshold is not None:
            result = [entry <= threshold for entry in result]
        return result if intermediate_results else result[-1]

    def conformance(self, generator, hypothesis):
        pass

    def dump_internal_data(self, path):
        self._monitor.dump_internal_data(path)


