param map = localPath('../../assets/maps/CARLA/Town05.xodr')
param carla_map = 'Town05'
param time_step = 1.0/10


model scenic.domains.driving.model


ego = new Car with behavior FollowLaneBehavior 
    


record distance from ego to ego.lane.centerline as CTE 
record ego.position as ego_position 
record ego.speed as speed 



