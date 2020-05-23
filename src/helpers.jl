# Decomposes a scene into smaller scenes made up of pairwise agents
function decompose_scene(scene::Scene, egoid::Int)
    scenes = Dict{Int, Scene}()
    ego = set_veh_id(get_by_id(scene, egoid), 2)
    for i=1:egoid-1
        if has_veh(i, scene)
            veh = set_veh_id(get_by_id(scene, i), 1)
            scenes[i] = Scene([veh, ego])
        end
    end
    scenes
end

# Functions for constructing vehicles
pos_in_Tint(index, s) = posg(Frenet(roadway_lane(Tint_roadway, index), s, 0.), Tint_roadway)

ego(id; s::Float64 = 40., vel::Float64 = 20.) = BlinkerVehicle(pos(5, s), vel, goals[5], 5, true, id, Tint_roadway)
left_straight(id; s::Float64 = 15., vel::Float64 = 19.) = BlinkerVehicle(pos(2, s), vel, goals[2], 2, false, id, Tint_roadway)
left_turnright(id; s::Float64 = 15., vel::Float64 = 19.) = BlinkerVehicle(pos(1, s), vel, goals[1], 1, true, id, Tint_roadway)
right_straight(id; s::Float64 = 30., vel::Float64 = 20.) = BlinkerVehicle(pos(3, s), vel, goals[3], 3, false, id, Tint_roadway)
right_turnleft(id; s::Float64 = 40., vel::Float64 = 14.) = BlinkerVehicle(pos(4, s), vel, goals[4], 4, true, id, Tint_roadway)

# Create a random IntelligentDriverModel
function random_IDM()
    headway_t = max(0.5, rand(rng, Normal(1.5, 0.5))) # desired time headway [s]
    v_des = max(15.0, rand(rng, Normal(20.0, 5.0))) # desired speed [m/s]
    s_min = max(1.0, rand(rng, Normal(5.0, 1.0))) # minimum acceptable gap [m]
    a_max = max(2.0, rand(rng, Normal(3.0, 1.0))) # maximum acceleration ability [m/s²]
    IntelligentDriverModel(T = headway_t, v_des = v_des, s_min = s_min, a_max = a_max)
end
