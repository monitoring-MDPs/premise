from multiprocessing import Queue, Process
import traceback
from typing import Optional
import scenic

from scenic.simulators.carla.simulator import CarlaSimulator
from tqdm import tqdm


def discr_color_get_level(value):
    if value <= 1 / 3:
        return "L"
    elif value <= 2 / 3:
        return "M"
    else:
        return "H"


def discr_color(simulation_result):
    r_level = discr_color_get_level(simulation_result["color"][0])
    g_level = discr_color_get_level(simulation_result["color"][1])
    b_level = discr_color_get_level(simulation_result["color"][2])
    return r_level + g_level + b_level


def discr_speed(simulation_result, x):
    if simulation_result["speed"][x][1] <= 2.5:
        return "slowest"
    elif simulation_result["speed"][x][1] <= 5:
        return "slow"
    elif simulation_result["speed"][x][1] <= 7.5:
        return "fast"
    else:
        return "fastest"


def discr_distance(simulation_result, x):
    if simulation_result["distance"][x][1] <= 10:
        return "d10"
    elif simulation_result["distance"][x][1] <= 20:
        return "d20"
    elif simulation_result["distance"][x][1] <= 30:
        return "d30"
    elif simulation_result["distance"][x][1] <= 40:
        return "d40"
    elif simulation_result["distance"][x][1] <= 50:
        return "d50"
    elif simulation_result["distance"][x][1] <= 60:
        return "d60"
    else:
        return "d70"


def discr_percived_distance(simulation_result, x):
    if simulation_result["percived_distance"][x][1] <= 10:
        return "pd10"
    elif simulation_result["percived_distance"][x][1] <= 20:
        return "pd20"
    elif simulation_result["percived_distance"][x][1] <= 30:
        return "pd30"
    elif simulation_result["percived_distance"][x][1] <= 40:
        return "pd40"
    elif simulation_result["percived_distance"][x][1] <= 50:
        return "pd50"
    elif simulation_result["percived_distance"][x][1] <= 60:
        return "pd60"
    else:
        return "pd70"


def discr_collision(simulation_result, x):
    return (
        (simulation_result["collision_0"][x][1] == True)
        or (simulation_result["collision_1"][x][1] == True)
        or (simulation_result["collision_2"][x][1] == True)
        or (simulation_result["collision_3"][x][1] == True)
    )


def sample(
    scenic_path: str,
    num_sim: int,
    time_steps: int,
    observation_prefix: Optional[list[tuple[str, str]]] = None,
    queue: Optional[Queue] = None,
    with_tqdm=False,
):
    if observation_prefix is not None and len(observation_prefix) > 0:
        params = {
            "color": observation_prefix[0][0],
            "trace": [int(d[2:]) for _, d in observation_prefix],
        }
    else:
        params = {}

    scenario = scenic.scenarioFromFile(
        scenic_path, model="scenic.simulators.carla.model", params=params, mode2D=True
    )

    records = []

    if with_tqdm:
        from tqdm import tqdm

        pb = tqdm(total=num_sim, desc="Simulations")

    while len(records) < num_sim:

        simulation = None

        scene, _ = scenario.generate(maxIterations=25000)

        try:
            simulator = CarlaSimulator(
                carla_map="Town01",
                map_path="./",
                timeout=60,
                render=0,  # type: ignore
            )

            simulation = simulator.simulate(scene, maxSteps=time_steps)
        except:
            traceback.print_exc()
            simulation = None
            pass

        if simulation:
            result = simulation.result
            if result is None:
                print("Simulation failed")
                continue

            if len(result.records["distance"]) >= time_steps:
                collision_detected = False
                trace = []
                for x in range(2, time_steps):
                    if not collision_detected:
                        if discr_collision(result.records, x) == False:
                            trace.append(
                                (
                                    (
                                        discr_speed(result.records, x),
                                        discr_distance(result.records, x),
                                        discr_color(result.records),
                                        discr_percived_distance(result.records, x),
                                    ),
                                    (
                                        discr_color(result.records),
                                        discr_percived_distance(result.records, x),
                                    ),
                                    discr_collision(result.records, x),
                                )
                            )
                        else:
                            collision_detected = True
                            trace.append((("collision"), ("collision"), True))
                    else:
                        trace.append((("collision"), ("collision"), True))

                if with_tqdm:
                    pb.update(1)
                records.append(trace)
                if queue is not None:
                    queue.put(trace)

        else:
            print("Restart Carla")

    return records


def sample_safe(
    scenic_path: str,
    num_sim: int,
    time_steps: int,
    observation_prefix: Optional[list] = None,
    with_tqdm=False,
):
    records = []
    queue = Queue()
    if with_tqdm:
        pb = tqdm(total=num_sim, desc="Simulations")
    while len(records) < num_sim:
        p = Process(
            target=sample,
            args=(scenic_path, num_sim, time_steps, observation_prefix, queue, False),
        )
        p.start()
        while p.is_alive():
            try:
                record = queue.get(timeout=1)
                records.append(record)
                if with_tqdm:
                    pb.update(1)
            except KeyboardInterrupt:
                p.terminate()
                p.join()
                raise KeyboardInterrupt
            except Exception as e:
                pass
        p.join()

    return records


def main(args):
    scenic_path = args.scenic_path
    num_sim = args.num_sim
    time_steps = args.num_steps
    if args.observation_prefix is not None:
        observation_prefix = eval(args.observation_prefix)
    else:
        observation_prefix = None

    records = sample_safe(
        scenic_path, num_sim, time_steps, observation_prefix, with_tqdm=True
    )
    print("Records:", records)
    print("Number of records:", len(records))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Data recording using records in Scenic.",
        usage="later",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ## Arguments
    parser.add_argument(
        "scenic_path", help="path to scenic file", metavar="scenic_path"
    )
    parser.add_argument(
        "--num-sim", help="number of simulations", type=int, default=1000
    )
    parser.add_argument(
        "--num-steps", help="number of steps per simulation", type=int, default=252
    )
    parser.add_argument(
        "--observation-prefix",
        help="observation prefix",
        type=str,
        default=None,
    )

    main_args = parser.parse_args()

    main(main_args)
