## Definition of new vehicle state, vehicle, scene, and action

# Define a new state for vehicles with blinkers
struct BlinkerState
    veh_state::VehicleState # position and velocity
    blinker::Bool # Whether or not the blinker is on
    goals::Vector{Int} # The list of possible goals that this vehicle can have
end

AutomotiveSimulator.posf(s::BlinkerState) = posf(s.veh_state)
AutomotiveSimulator.posg(s::BlinkerState) = posg(s.veh_state)
AutomotiveSimulator.vel(s::BlinkerState)  = vel(s.veh_state)
AutomotiveSimulator.velf(s::BlinkerState) = velf(s.veh_state)
AutomotiveSimulator.velg(s::BlinkerState) = velg(s.veh_state)

# Constructor for making a blinker vehicle
function BlinkerVehicle(posG::VecSE2, v::Float64, goals::Vector{Int}, goal_lane::Int, blinker, id, roadway::Roadway)
    vs = VehicleState(posG, roadway, v)
    bv = Entity(BlinkerState(vs, blinker, goals), VehicleDef(), id)
    set_veh_lane(bv, goal_lane, roadway)
end

# How to render the blinker vehicle.
function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, veh::Entity{BlinkerState, VehicleDef, Int64})
    reg_veh = Entity(veh.state.veh_state, veh.def, veh.id)
    add_renderable!(
        rendermodel, FancyCar(car=reg_veh)
    )
    li = laneid(veh)
    add_renderable!(
        rendermodel, BlinkerOverlay(on = veh.state.blinker, veh = reg_veh, right=T_int_signal_right[li])
    )
    return rendermodel
end

# decomposes the scene into the vehicles indicated by the indices
function decompose_scene(scene::Scene, ids::Vector{Int})
    new_scene = Scene(typeof(get_by_id(scene, ids[1])))
    count = 1
    for i in ids

        count += 1
    end
    new_scene
end


## Definition of LaneFollowingAccelBlinker action with helpers and propagate

# Define a new action that can set laneid and blinker state
@with_kw struct LaneFollowingAccelBlinker
    a::Float64 = 0
    da::Float64 = 0
    toggle_goal::Bool = false
    toggle_blinker::Bool = false
end

# The function that propogates the new action
function AutomotiveSimulator.propagate(veh::Entity, action::LaneFollowingAccelBlinker, roadway::Roadway, Δt::Float64)
    # set the new goal
    if action.toggle_goal
        curr_index = findfirst(veh.state.goals .== laneid(veh))
        @assert !isnothing(curr_index)
        new_goal = veh.state.goals[curr_index % length(veh.state.goals) + 1]
        if can_have_goal(veh, new_goal, roadway)
            veh = set_veh_lane(veh, new_goal, roadway)
        end
    end

    # Update the kinematics of the vehicle (don't allow v < 0)
    a = action.a + action.da
    ds = vel(veh)
    Δs = max(0,ds*Δt + 0.5*a*Δt*Δt)
    v₂ = max(0,ds + a*Δt)

    roadind = move_along(posf(veh).roadind, roadway, Δs)
    posG = roadway[roadind].pos
    posF = Frenet(roadind, roadway, t=posf(veh).t, ϕ=posf(veh).ϕ)

    # Set the blinker state and return
    new_blink = action.toggle_blinker ? !veh.state.blinker : veh.state.blinker
    BlinkerState(VehicleState(posG, posF, v₂), new_blink, veh.state.goals)
end


## Definition of the T-Intersection DriverModel with observe! and rand()

# Define a driving model for a T-intersection IDM model
@with_kw mutable struct TIDM <: DriverModel{LaneFollowingAccelBlinker}
    idm::IntelligentDriverModel = IntelligentDriverModel()

    # Defines the stochastic actions of the agents
    ttc_threshold = 5 # threshold through intersection

    # The next action that the model will do
    next_action::LaneFollowingAccelBlinker = LaneFollowingAccelBlinker()

    # Describes the intersection and rules of the road
    yields_way::Dict{Int64, Vector{Int64}} = Dict() # Lane priorities
    intersection_enter_loc::Dict{Int64, VecSE2} = Dict() # Entry location of intersection
    intersection_exit_loc::Dict{Int64, VecSE2} = Dict()  # Exit location of intersection
    goals::Dict{Int64, Vector{Int64}} = Dict() # Possible goals of each lane
    should_blink::Dict{Int64, Bool} = Dict()  # Wether or not the blinker should be on
end

# Sample an action from TIDM model
function Base.rand(rng::AbstractRNG, model::TIDM)
    LaneFollowingAccelBlinker(model.idm.a, model.next_action.da, model.next_action.toggle_goal, model.next_action.toggle_blinker)
 end

# Get the belief of tha lane of the specificed vehicle
function lane_belief(veh::Entity, model::TIDM, roadway::Roadway)
    possible_lanes = model.goals[laneid(veh)]
    @assert length(possible_lanes) == 2

    possible_lanes = possible_lanes[[can_have_goal(veh, l, roadway) for l in possible_lanes]]
    length(possible_lanes) == 1 && return possible_lanes[1]

    blinker_match = findfirst([veh.state.blinker == model.should_blink[l] for l in possible_lanes])
    if !isnothing(blinker_match)
        return possible_lanes[blinker_match]
    else
        return model.goals[laneid(veh)][1]
    end
end

# Observe function for TIDM
function AutomotiveSimulator.observe!(model::TIDM, scene::Scene, roadway::Roadway, egoid::Int64)
    # Pull out the necessary quantities
    vehicle_index = findfirst(egoid, scene)
    ego = scene[vehicle_index]
    li = laneid(ego)

    # Compute ego headway to intersection and time to cross
    int_headway = distance_to_point(ego, roadway, model.intersection_enter_loc[li])
    time_to_cross = time_to_cross_distance_const_acc(ego, model.idm, distance_to_point(ego, roadway, model.intersection_exit_loc[li]))

    # Get headway to the forward car
    fore = find_neighbor(scene, roadway, ego, targetpoint_ego = VehicleTargetPointFront(), targetpoint_neighbor = VehicleTargetPointRear())

    v_oth, for_car_headway = NaN, NaN
    if fore.ind != nothing
        v_oth = vel(scene[fore.ind].state)
        for_car_headway = fore.Δs
    end

    # Check to see if ego car has right of way
    has_right_of_way = true
    lanes_to_yield_to = model.yields_way[li]
    vehicles_to_yield_to = []
    for (i,veh) in enumerate(scene)
        if veh.id != egoid && lane_belief(veh, model, roadway) in lanes_to_yield_to
            has_right_of_way = false
            push!(vehicles_to_yield_to, veh)
        end
    end
    v_ego = vel(ego.state)

    # TODO: Reasoning to take place if you enter an intersection
    # if i_headway <-r
        # has_right_of_way = true
    # end

    next_idm = track_longitudinal!(model.idm, v_ego, v_oth, for_car_headway)
    # If the vehicle does not have right of way then stop before the intersection
    if !has_right_of_way
        # Compare ttc
        exit_time = [time_to_cross_distance_const_vel(veh, distance_to_point(veh, roadway, model.intersection_exit_loc[laneid(veh)])) for veh in vehicles_to_yield_to]
        enter_time = [time_to_cross_distance_const_vel(veh, distance_to_point(veh, roadway, model.intersection_enter_loc[laneid(veh)])) for veh in vehicles_to_yield_to]
        cars_to_care = exit_time .> 0
        exit_time = exit_time[cars_to_care] # remove cars that are already crossed
        enter_time = enter_time[cars_to_care] # remove cars that are already crossed
        if !all((enter_time .> time_to_cross) .| (exit_time .+ model.ttc_threshold .< time_to_cross))
            # yield to oncoming traffic
            # If there isn't a leading car, or if it is past the intersection, use intersection point, otherwise use car
            if isnan(for_car_headway) || (int_headway > 0 && int_headway < for_car_headway)
                next_idm = track_longitudinal!(model.idm, v_ego, 0., int_headway)
            end
        end
    end
    model.idm = next_idm
    model
end


## Definition of helper functions for managing vehicles on the road
AutomotiveSimulator.get_lane(roadway::Roadway, state::BlinkerState) = roadway[posf(state).roadind.tag]

# Get the laneid of a vehicle
laneid(veh::Entity) = posf(veh).roadind.tag.segment

# Returns a vehicle with the same state, but projected onto the desired lane
function set_veh_lane(veh::Entity, laneid::Int, roadway::Roadway)
    desired_lane = roadway[laneid].lanes[1]
    posF = Frenet(posg(veh), desired_lane, roadway)
    Entity(BlinkerState(VehicleState(posF, roadway, vel(veh)), veh.state.blinker, veh.state.goals), veh.def, veh.id)
end

# Returns a new vehicle that is a copy of the provided vehicle except with the specified id
set_veh_id(veh::Entity, id::Int) = Entity(veh.state, veh.def, id)

# Check whether a car is allowed to switch goals (requires posF.t to be small)
function can_have_goal(veh::Entity, goal::Int, roadway::Roadway; ϵ = 1e-6)
    desired_lane = roadway[goal].lanes[1]
    posF = Frenet(posg(veh), desired_lane, roadway)
    return abs(posF.t) <= ϵ
end

# Computes the time it takes to cover a given distance, assuming the maximum acceleration of the provided idm
function time_to_cross_distance_const_acc(veh::Entity, idm::IntelligentDriverModel, ds::Float64)
    v = vel(veh)
    d = v^2 + 2*idm.a_max*ds
    d < 0 && return 0 # We have already passed the point we are trying to get to
    vf = min(idm.v_des, sqrt(d))
    2*ds/(vf + v)
end

# Compute the time it takes to travel a given distance at the current vehicle velocity
time_to_cross_distance_const_vel(veh::Entity, ds::Float64) = ds / vel(veh)

# Computes the distance between the vehicle and the provided point along the roadway
# If the pt is not on the roadway then it is the distance to the projection onto the road
function distance_to_point(veh::Entity, roadway::Roadway, pt::VecSE2)
    s0 = posf(veh).s
    lane = get_lane(roadway, veh.state.veh_state)
    s_end = Frenet(pt, lane, roadway).s
    s_end - s0
end

# computes the closest distance between the ego vehicle and another vehicle on the road
function min_dist(scene, egoid)
    minval = 10000
    (length(scene) == 0 || !has_veh(egoid, scene)) && return minval
    ego_pos = posg(get_by_id(scene, egoid).state)
    for (i,veh) in enumerate(scene)
        if veh.id != egoid
            pos = posg(veh)
            dist = sqrt((pos.x - ego_pos.x)^2 + (pos.y - ego_pos.y)^2)
            if dist < minval
                minval = dist
            end
        end
    end
    minval
end

# Define a callback to remove cars that have reached the end of the lane
struct CleanSceneCallback
    egoid
end

function has_veh(id, scene)
    has_veh = false
    for (i,veh) in enumerate(scene)
        if veh.id == id
            has_veh = true
        end
    end
    return has_veh
end

# Check if the ego vehicle is in collision with another car
function ego_collides(egoid, scene)
    if !has_veh(egoid, scene)
        return false
    end

    ego = get_by_id(scene, egoid)
    for (i,veh) in enumerate(scene)
        if egoid != veh.id && collision_checker(ego, veh)
            return true
        end
    end
    false
end


# Check if any two vehicles collide
function any_collides(scene)
    for (i,veh) in enumerate(scene)
        for (j,veh2) in enumerate(scene)
            j <= i && continue
            collision_checker(veh, veh2) && return true
        end
    end
    false
end

# Check if a given car has reached the end of the roadway
function end_of_road(veh, roadway)
    s = posf(veh).s
    lane = get_lane(roadway, veh)
    s_end = lane.curve[end].s
    return s >= s_end
end

# Removes cars that have reached the end of the lane
function AutomotiveSimulator.run_callback(c::CleanSceneCallback, scenes::Vector{Scene}, actions::Nothing, roadway::R, models::Dict{I,M}, tick::Int) where {F,I,R,M<:DriverModel}
    for (i,veh) in enumerate(scenes[tick])
        if end_of_road(veh, roadway)
           deleteat!(scenes[tick], findfirst(veh.id, scenes[tick]))
       end
    end
    return ego_collides(scenes[tick], c.egoid)
end

