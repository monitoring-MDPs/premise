import scenic

from scenic.simulators.carla.simulator import CarlaSimulator


def discr_color_get_level(value): 
    if value <= 1/3: 
         return 'L'
    elif value <= 2/3: 
        return 'M'
    else:
        return 'H'

def discr_color(simulation_result): 
    r_level = discr_color_get_level(simulation_result['color'][0])
    g_level = discr_color_get_level(simulation_result['color'][1])
    b_level = discr_color_get_level(simulation_result['color'][2])
    return r_level + g_level + b_level

        
def discr_speed(simulation_result,x): 
    if simulation_result['speed'][x][1] <= 2.5: 
        return 'slowest'
    elif simulation_result['speed'][x][1] <= 5: 
        return 'slow'
    elif simulation_result['speed'][x][1] <= 7.5: 
        return 'fast'
    else: 
        return 'fastest'

def discr_distance(simulation_result,x): 
    if simulation_result['distance'][x][1] <= 10: 
        return 'd10'
    elif simulation_result['distance'][x][1] <= 20: 
        return 'd20'
    elif simulation_result['distance'][x][1] <= 30: 
        return 'd30'
    elif simulation_result['distance'][x][1] <= 40: 
        return 'd40'
    elif simulation_result['distance'][x][1] <= 50: 
        return 'd50'
    elif simulation_result['distance'][x][1] <= 60: 
        return 'd60'
    else: 
        return 'd70'

def discr_percived_distance(simulation_result,x): 
    if simulation_result['percived_distance'][x][1] <= 10: 
        return 'pd10'
    elif simulation_result['percived_distance'][x][1] <= 20: 
        return 'pd20'
    elif simulation_result['percived_distance'][x][1] <= 30: 
        return 'pd30'
    elif simulation_result['percived_distance'][x][1] <= 40: 
        return 'pd40'
    elif simulation_result['percived_distance'][x][1] <= 50: 
        return 'pd50'
    elif simulation_result['percived_distance'][x][1] <= 60: 
        return 'pd60'
    else: 
        return 'pd70'

def discr_collision(simulation_result,x): 
    return simulation_result['collision'][x][1]

def sample(scenic_path, num_sim, time_steps): 
    scenario = scenic.scenarioFromFile(scenic_path,
                                   model='scenic.simulators.carla.model',
                                   mode2D=True)

    records = []

    while len(records)<num_sim:


        simulation = None
    
        scene, _ = scenario.generate()

        try: 
            simulator = CarlaSimulator(carla_map = 'Town01', 
                                        map_path = './maps/Town01.xodr',
                                        timeout = 60,
                                        render = 1)

            simulation = simulator.simulate(scene, maxSteps=time_steps)     
        except: 
            simulation = None
            pass
   
        if simulation:
            result = simulation.result

            if len(result.records["distance"]) >= time_steps: 
                records.append([
                        (discr_color(result.records), discr_speed(result.records, x), discr_distance(result.records, x), discr_percived_distance(result.records, x), discr_collision(result.records, x))
                        for x in range(2, time_steps)]
                )
        else:
            print('Restart Carla')
    return records
    
def main(args):
    scenic_path = args.scenic_path
    num_sim = args.num_sim
    time_steps = args.num_steps
    rounds = args.num_rounds

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Data recording using records in Scenic.',usage='later', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Arguments 
    parser.add_argument('scenic_path', help='path to scenic file', metavar='scenic_path')
    parser.add_argument('--num_sim', help='number of simulations', type=int, default=1000)
    parser.add_argument('--num_steps', help='number of steps per simulation',type=int,default=202)

    main_args = parser.parse_args()

    main(main_args)
