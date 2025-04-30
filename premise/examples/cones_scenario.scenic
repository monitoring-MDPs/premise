param timeout = 60
param map = localPath('assets/maps/Town01.xodr')
param carla_map = 'Town01'
param render = 1

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

ego = new ConesCar at start, with behavior EgoBehavior()


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

require broken_car in ego.lane

record ego.speed as speed 
record initial broken_car.color as color 
record ego.current_front_img as imgs 
record ego.distanceToClosest(Object) as distance
record ego intersects broken_car as collision
record ego.getDistanceCNN() as percived_distance


