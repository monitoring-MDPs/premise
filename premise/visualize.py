"""
Visualises policy monitoring (see policy_monitoring.py) on a simulated trace, using gridstorm
(https://github.com/sjunges/gridworld-by-storm). Install it with `uv sync --group viz`.

Every frame shows the true state of the simulation together with the monitor's estimates after observing the trace so far:
the posterior over the hypotheses, the risk under every hypothesis and the weighted risk.
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")  # Render without a display.

from gridstorm.annotations import ProgramAnnotation  # noqa: E402
from gridstorm.plotter import Plotter, colors, data_colors  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from gridstorm.trace import Trace  # noqa: E402
import stormpy as sp  # noqa: E402
import stormpy.simulator  # noqa: E402

from premise import policy_monitoring as pm  # noqa: E402

logger = logging.getLogger(__name__)

# How gridstorm finds the agents, the grid size and the special locations in the models of the benchmarks.
grid_annotations = {
    # The ego vehicle moves randomly; the drone (drawn as adversary) is the agent whose policy is monitored.
    "evade": {
        "ego-xvar-module": "ego",
        "ego-xvar-name": "ax",
        "ego-yvar-module": "ego",
        "ego-yvar-name": "ay",
        "adv-xvar-module": "drone",
        "adv-xvar-name": "dx",
        "adv-yvar-module": "drone",
        "adv-yvar-name": "dy",
        "xmax-constant": "xMAX",
        "ymax-constant": "yMAX",
        "target-label": "goal",
        "traps-label": None,
        "ego-radius-constant": "RADIUS",
        "adv-goals-label": "targets",
    },
    # The rover is the agent whose policy is monitored; both refuelling stations are drawn as landmarks.
    "refuel": {
        "ego-xvar-module": "rover",
        "ego-xvar-name": "ax",
        "ego-yvar-module": "rover",
        "ego-yvar-name": "ay",
        "xmax-constant": "axMAX",
        "ymax-constant": "ayMAX",
        "target-label": "goal",
        "traps-label": "traps",
        "landmarks": "stationvisit",
        "resource-module": "tank",
        "resource-variable": "fuel",
        "resource-name": "fuel",
        "resource-maximum-constant": "fuelCAP",
    },
}



def _cell(kind):
    """Style of a grid cell as gridstorm colours it (kind is a key of gridstorm.plotter.data_colors)."""
    return dict(facecolor=colors.colors[data_colors[kind]], edgecolor="black", linewidth=0.3)


# Style of the agents as drawn by gridstorm.
_EGO = dict(facecolor="b", edgecolor="b")
_EGO_VIEW = dict(facecolor="b", edgecolor="b", alpha=0.15, hatch="o")
_ADVERSARY = dict(facecolor="sienna", edgecolor="saddlebrown")

# The legend of every benchmark: what the colours in the grid mean.
legends = {
    "evade": [
        ("ego vehicle", _EGO),
        ("ego view radius", _EGO_VIEW),
        ("ego goal", _cell("GOAL")),
        ("drone (agent)", _ADVERSARY),
        ("drone targets A, B", _cell("ADVGOAL")),
    ],
    "refuel": [
        ("rover (agent)", _EGO),
        ("refuelling stations A, B", _cell("LANDMARK")),
        ("obstacle", _cell("TRAP")),
        ("goal", _cell("GOAL")),
    ],
}

# Locations to mark with a text in the grid: the text and the model constants of its x and y coordinate.
markers = {
    "evade": [("A", "TAX", "TAY"), ("B", "TBX", "TBY")],
    "refuel": [("A", "rfAx", "rfAy"), ("B", "rfBx", "rfBy")],
}


class MonitorPlotter(Plotter):
    """
    A gridstorm plotter that additionally shows the estimates of the monitor in every frame.
    """

    def __init__(self, program, annotation, model, estimates, names, legend=(), markers=()):
        # estimates[i] belongs to the i-th state of the trace; it is None for the initial state (no observations yet).
        self._estimates = estimates
        self._names = names
        # Pairs of a label and the style of its patch.
        self._legend = legend
        # Triples of a text and the constants of the x and y coordinate of the cell to mark with it.
        self._markers = markers
        super().__init__(program, annotation, model)

    def _init_fig(self):
        super()._init_fig()
        # Leave room below the grid for the estimates and the legend.
        self._fig.subplots_adjust(bottom=0.2)
        if self._legend:
            handles = [Patch(label=label, **style) for label, style in self._legend]
            self._fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.86, 0.0), fontsize=8, framealpha=0.9)
        for text, x_constant, y_constant in self._markers:
            x = self._program.get_constant(x_constant).definition.evaluate_as_int()
            y = self._program.get_constant(y_constant).definition.evaluate_as_int()
            # In the corner of the cell, such that agents in the middle of the cell stay visible.
            self._ax.text(x + 0.08, y + 0.08, text, fontsize=11, fontweight="bold", verticalalignment="top", zorder=10)

    def render(self, snapshot, show_frame_count=None, show=False):
        super().render(snapshot, show_frame_count=show_frame_count, show=show)
        estimate = self._estimates[snapshot.index]
        if estimate is None:
            text = "no observations yet (prior)"
        else:
            lines = [f"weighted risk {float(estimate.risk):.4f}"]
            for name in self._names:
                risk = estimate.conditional_risk[name]
                risk_text = "-" if risk is None else f"{float(risk):.4f}"
                lines.append(f"P({name}) {float(estimate.posterior[name]):.3f}   risk({name}) {risk_text}")
            text = "\n".join(lines)
        txt = self._fig.text(0.01, 0.01, text, fontsize=9, family="monospace", verticalalignment="bottom")
        self._tmp_objects.append(txt)


def simulate(hmm, length, seed):
    """
    Simulates the HMM and records the (full) states as a gridstorm trace.
    """
    simulator = sp.simulator.create_simulator(hmm, seed)
    simulator.set_full_observability(True)
    trace = Trace()
    state, _, _ = simulator.restart()
    trace.append_state(state)
    for _ in range(length):
        actions = simulator.available_actions()
        # The HMM has a single action in every state.
        state, _, _ = simulator.step(actions[0])
        trace.append_available_actions(actions)
        trace.append_considered_actions(actions)
        trace.append_action(actions[0])
        trace.append_state(state)
    actions = simulator.available_actions()
    trace.append_available_actions(actions)
    trace.append_considered_actions(actions)
    trace.append_action(None)
    return trace


def main():
    parser = argparse.ArgumentParser(description="Visualise policy monitoring on a simulated trace.")
    parser.add_argument("--benchmark", choices=list(pm.benchmarks), default="evade")
    parser.add_argument(
        "--true-hypothesis",
        help="The hypothesis whose HMM generates the trace (default: the first hypothesis of the benchmark).",
    )
    parser.add_argument("--trace-length", default=20, type=int, help="How long should the trace be?")
    parser.add_argument("--seed", default=0, type=int, help="Seed for the simulator.")
    parser.add_argument("--float", action="store_false", dest="exact", help="Use floating point numbers instead of exact arithmetic.")
    parser.add_argument("--output", help="Output file, .gif (needs ImageMagick) or .mp4 (needs FFmpeg).")
    args = parser.parse_args()

    benchmark = pm.benchmarks[args.benchmark]
    hypotheses = pm.build_hypotheses(benchmark, args.exact)
    names = [hypothesis.name for hypothesis in hypotheses]
    true_name = args.true_hypothesis if args.true_hypothesis is not None else names[0]
    if true_name not in names:
        parser.error(f"Unknown hypothesis {true_name}, choose from {names}.")
    hmm = hypotheses[names.index(true_name)].hmm

    # Simulate the true hypothesis and monitor the observations of the visited states.
    trace = simulate(hmm, args.trace_length, args.seed)
    observations = [hmm.observations[state] for state in trace._states]
    monitor = pm.PolicyMixtureMonitor(hypotheses)
    monitor.initialize(observations[0])
    estimates = [None] + [monitor.step(observation) for observation in observations[1:]]

    program, _ = pm.load_program(benchmark)
    annotation = ProgramAnnotation(dict(grid_annotations[args.benchmark]))
    plotter = MonitorPlotter(program, annotation, hmm, estimates, names, legends[args.benchmark], markers[args.benchmark])
    plotter.set_title(f"{args.benchmark}: trace of {true_name}")
    output = args.output if args.output is not None else f"{args.benchmark}-{true_name}-{args.seed}.gif"
    plotter.record(output, trace)
    print(f"Written {output}")


if __name__ == "__main__":
    main()
