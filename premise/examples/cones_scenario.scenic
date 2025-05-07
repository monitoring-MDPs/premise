param timeout = 60
param map = localPath('../../assets/maps/CARLA/Town01.xodr')
param carla_map = 'Town01'
param render = 0

param trace = []
param color = None

import carla

from premise.carla.ConesCar import ConesCar
model scenic.simulators.carla.model

param weather = 'ClearNoon'


behavior EgoBehavior():
    try:
        do FollowLaneBehavior()
    interrupt when ego.getDistanceCNN() <= 10:   
        take SetThrottleAction(0), SetBrakeAction(1)


lane = Uniform(*network.lanes)
start = new OrientedPoint on lane.centerline

ego = new ConesCar at start, with behavior EgoBehavior() #COLOR 


blockageSite = new OrientedPoint ahead of ego by Range(50,65)

spot1 = new OrientedPoint left of blockageSite by Range(0, 1)
cone1 = new Cone at spot1, facing Range(0, 360) deg, 
    with blueprint "static.prop.constructioncone"

spot2 = new OrientedPoint ahead of spot1 by Range(-2, -1) @ Range(1, 4)
cone2 = new Cone at spot2,  facing Range(0, 360) deg, 
    with blueprint "static.prop.constructioncone"

cone3 = new Cone ahead of spot1 by Range(2, 3) @ Range(-0.5, 0.5),
    with blueprint "static.prop.constructioncone"


badAngle = Uniform(1.0, -1.0) * Range(15, 70) deg

broken_car  = new ConesCar ahead of blockageSite by Range(0, 1), facing badAngle relative to roadDirection

monitor ConditionOnTrace():
    step = -2
    while step < len(globalParameters.trace):
        if step < 0:
            pass
        else:
            dist = ego.getDistanceCNN()
            print(dist)
            if globalParameters.trace[step] == 70:
                require 60 < dist
            else:
                require globalParameters.trace[step] - 10 < dist <= globalParameters.trace[step]
        step += 1
        wait

# monitor ConditionOnColor():
#     if globalParameters.color is not None:
#         for i, c in enumerate(globalParameters.color):
#             if c == "L":
#                 require broken_car.color[i] < 0.33
#             elif c == "M":
#                 require broken_car.color[i] >= 0.33
#                 require broken_car.color[i] < 0.67
#             else:
#                 require broken_car.color[i] >= 0.67
#     wait

require monitor ConditionOnTrace()
# require monitor ConditionOnColor()
require globalParameters.color is None or
    globalParameters.color[0] == "L" and broken_car.color[0] < 0.33 or
    globalParameters.color[1] == "L" and broken_car.color[1] < 0.33 or
    globalParameters.color[2] == "L" and broken_car.color[2] < 0.33 or
    globalParameters.color[0] == "M" and 0.33 <= broken_car.color[0] < 0.67 or
    globalParameters.color[1] == "M" and 0.33 <= broken_car.color[1] < 0.67 or
    globalParameters.color[2] == "M" and 0.33 <= broken_car.color[2] < 0.67 or
    globalParameters.color[0] == "H" and 0.67 <= broken_car.color[0] or
    globalParameters.color[1] == "H" and 0.67 <= broken_car.color[1] or
    globalParameters.color[2] == "H" and 0.67 <= broken_car.color[2]

require broken_car in ego.lane

record ego.speed as speed 
record initial broken_car.color as color 
record ego.current_front_img as imgs 
record ego.distanceToClosest(Object) as distance
record ego intersects broken_car as collision_0
record ego intersects cone1 as collision_1
record ego intersects cone2 as collision_2
record ego intersects cone3 as collision_3
record ego.getDistanceCNN() as percived_distance


