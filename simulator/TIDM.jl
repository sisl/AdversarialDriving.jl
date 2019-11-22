using AutomotiveDrivingModels
using AutoViz
using Distributions
using Parameters
using LinearAlgebra
using Random


# Definition of new vehicle state, vehicle, scene, and action
############################################################################
############################################################################

# Define a new state for vehicles with blinkers
struct BlinkerState
    veh_state::VehicleState
    blinker::Bool
end

AutomotiveDrivingModels.posf(s::BlinkerState) = posf(s.veh_state)
AutomotiveDrivingModels.posg(s::BlinkerState) = s.veh_state.posG
vel(s::BlinkerState) = s.veh_state.v

# Define a new Vehicle with the BlinkerVehicle
BlinkerVehicle = Entity{BlinkerState, VehicleDef, Int64}

function BV(posG::VecSE2, v::Float64, goal_lane::Int, blinker, id, roadway::Roadway)
    vs = VehicleState(posG, roadway, v)
    bv = BlinkerVehicle(BlinkerState(vs, blinker), VehicleDef(), id)
    set_veh_lane(bv, goal_lane, roadway)
end

# Define a scene that consists of Blinkervehicles
const BlinkerScene = Frame{BlinkerVehicle}
BlinkerScene(n::Int=100) = Frame(BlinkerVehicle, n)

# Convert BlinkerVehicle to Vehicle
AutomotiveDrivingModels.Vehicle(b::BlinkerVehicle) = Vehicle(b.state.veh_state, b.def, b.id)

# Convert BlinkerScene to Scene
function AutomotiveDrivingModels.Scene(bs::BlinkerScene)
    s = Scene()
    for (i,veh) in enumerate(bs)
        push!(s, Vehicle(veh))
    end
    s
end

# Instruction on how to render a BlinkerVehicle
AutoViz.render!(r::RenderModel, veh::BlinkerVehicle, c::Colorant) = render!(r, Vehicle(veh), c)



# Definition of LaneFollowingAccelBlinker action with helpers and propagate
############################################################################
############################################################################

# Define a new action that can set laneid and blinker state
struct LaneFollowingAccelBlinker
    a::Float64
    da::Float64
    laneid::Int
    blinker::Bool
end

# The function that propogates the new action
function AutomotiveDrivingModels.propagate(veh::BlinkerVehicle, action::LaneFollowingAccelBlinker, roadway::Roadway, Δt::Float64)
    # Set the vehicle lane
    if action.laneid != 0 && can_have_goal(veh, action.laneid, roadway)
        veh = set_veh_lane(veh, action.laneid, roadway)
    end

    # Update the kinematics of the vehicle (don't allow v < 0)
    a = action.a + action.da
    ds = vel(veh.state)
    Δs = max(0,ds*Δt + 0.5*a*Δt*Δt)
    v₂ = max(0,ds + a*Δt)

    roadind = move_along(posf(veh.state).roadind, roadway, Δs)
    posG = roadway[roadind].pos
    posF = Frenet(roadind, roadway, t=posf(veh.state).t, ϕ=posf(veh.state).ϕ)

    # Set the blinker state and return
    BlinkerState(VehicleState(posG, posF, v₂), action.blinker)
end


# Definition of the T-Intersection DriverModel with observe! and rand()
############################################################################
############################################################################

# Define a driving model for a T-intersection IDM model
@with_kw mutable struct TIDM <: DriverModel{LaneFollowingAccelBlinker}
    idm::IntelligentDriverModel = IntelligentDriverModel()

    # Defines the stochastic actions of the agents
    ttc_threshold = 5 # threshold through intersection
    p_change_goal = 1e-4 # Probability of changing goal at a given timestep
    p_wrong_signal = 1e-4 # Probability of having the incorrect signal on
    da_dist::Normal = Normal(0,1)# Distributions over acc
    goal_dist::Categorical = Categorical(6) # Distribution over changing goals
    blinker_dist::Bernoulli = Bernoulli(0) # Distribution over toggling signal

    # The members below here are for control of AST
    force_action::Bool = false # Whether or not we should use the stored actions
    da_force::Float64 = 0. # The acceleration to apply
    goal_force::Int64 = false # Whether to toggle a goal
    blinker_force::Bool = false # Whether to toggle the turn signal

    # Describes the intersection and rules of the road
    yields_way::Dict{Int, Vector{Int}} = Dict() # Lane priorities
    intersection_enter_loc::Dict{Int, VecSE2} = Dict() # Entry location of intersection
    intersection_exit_loc::Dict{Int, VecSE2} = Dict()  # Exit location of intersection
    goals::Dict{Int64, Vector{Int64}} = Dict() # Possible goals of each lane
    should_blink::Dict{Int64, Array{Int}} = Dict()  # Wether or not the blinker should be on
end

# Easy generation function for getting a driving that is controllable by AST
function generate_TIDM_AST(intersection_enter_loc, intersection_exit_loc, goals, should_blink)
    TIDM(   force_action = true,
            intersection_enter_loc = intersection_enter_loc,
            intersection_exit_loc = intersection_exit_loc,
            goals = goals
            should_blink = should_blink,
            )
end

# Make a copy of an existing model, while replacing the id, and probabilities
function generate_TIDM_AST(template::TIDM, p_wrong_signal, σ2a)
    TIDM(   p_wrong_signal = p_wrong_signal,
            da_dist = Normal(0,σ2a),
            force_action = template.force_action,
            intersection_enter_loc = template.intersection_enter_loc,
            intersection_exit_loc = template.intersection_exit_loc,
            goals = template.goals
            should_blink = template.should_blink,
            )
end

# Get the probability density of the specified action
function get_actions_logpd(model::TIDM, action::LaneFollowingAccelBlinker)
    a_pd = logpdf(model.da_dist, action.da)
    goal_pm = logpdf(model.goal_dist, action.laneid)
    blinker_pm = logpdf(model.blinker_dist, action.blinker)
    a_pd + goal_pm + blinker_pm
end

# Gets a random action from the model, ignoring the force flag
function random_action(model::TIDM, rng::AbstractRNG = Random.GLOBAL_RNG)
    da = rand(rng, model.da_dist)
    g = rand(rng, model.goal_dist)
    b = rand(rng, model.blinker_dist)
    LaneFollowingAccelBlinker(model.idm.a, da, g, b)
end

# Sample an action from TIDM model
function Base.rand(rng::AbstractRNG, model::TIDM)
    if model.force_action # Use the forced actions
        LaneFollowingAccelBlinker(model.idm.a, model.da_force, model.goal_force, model.blinker_force)
    else # sample actions from the distributions
        random_action(model, rng)
    end
 end

# Name of the driving model
AutomotiveDrivingModels.get_name(::TIDM) = "T-Intersection IDM"

function lane_belief(veh::BlinkerVehicle, model::TIDM, roadway::Roadway)
    possible_lanes = model.goals[laneid(veh)]
    @assert length(possible_lanes) == 2

    allowed_lanes = [can_have_goal(veh, l, roadway) for l in possible_lanes]

    if any(allowed_lanes) && !all(allowed_lanes)
        pl = possible_lanes[findfirst(allowed_lanes)]
    end

    blinker_match = [veh.state.blinker == should_blink[l] for l in possible_lanes]
    if findfirst(blinker_match) != nothing
        return possible_lanes[findfirst(blinker_match)]
    else
        return rand(possible_lanes)
    end
end

# Observe function for TIDM
function AutomotiveDrivingModels.observe!(model::TIDM, scene::BlinkerScene, roadway::Roadway, egoid::Int64)
    # Pull out the necessary quantities
    vehicle_index = findfirst(egoid, scene)
    ego = scene[vehicle_index]
    li = laneid(ego)

    # Update distributions
    goal_vec = zeros(6)
    ego_goals = model.goals[li]
    @assert li in ego_goals
    if length(ego_goals) == 1
        goal_vec[ego_goals[1]] = 1
    else
        p_other = model.p_change_goal / (length(ego_goals) - 1)
        p_goal = 1 - model.p_change_goal
        goal_vec[li] = p_goal
        goal_vec[ego_goals[ego_goals .!= li]] .= p_other
    end
    model.goal_dist = Categorical(goal_vec)

    p_blink = (1-model.p_wrong_signal)*model.should_blink[li] + model.p_wrong_signal*(!model.should_blink[li])
    model.blinker_dist = Bernoulli(p_blink) # Distribution over signal


    # Compute ego headway to intersection and time to cross
    int_headway = distance_to_point(ego, roadway, model.intersection_enter_loc[li])
    time_to_cross = time_to_cross_distance_const_acc(ego, model.idm, distance_to_point(ego, roadway, model.intersection_exit_loc[li]))

    # Get headway to the forward car
    fore = get_neighbor_fore_along_lane(Scene(scene), vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())

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
        if lane_belief(veh, model, roadway) in lanes_to_yield_to
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


# Definition of helper functions for managing vehicles on the road
############################################################################
############################################################################

# Get the laneid of a vehicle
laneid(veh::BlinkerVehicle) = posf(veh.state).roadind.tag.segment

# Returns a vehicle with the same state, but projected onto the desired lane
function set_veh_lane(veh::BlinkerVehicle, laneid::Int, roadway::Roadway)
    desired_lane = roadway[laneid].lanes[1]
    posF = Frenet(posg(veh.state), desired_lane, roadway)
    BlinkerVehicle(BlinkerState(VehicleState(posF, roadway, vel(veh.state)), veh.state.blinker), veh.def, veh.id)
end

# Check whether a car is allowed to switch goals (requires posF.t to be small)
function can_have_goal(veh::BlinkerVehicle, goal::Int, roadway::Roadway; ϵ = 1e-6)
    desired_lane = roadway[goal].lanes[1]
    posF = Frenet(posg(veh.state), desired_lane, roadway)
    return abs(posF.t) <= ϵ
end

# Computes the time it takes to cover a given distance, assuming the maximum acceleration of the provided idm
function time_to_cross_distance_const_acc(veh::BlinkerVehicle, idm::IntelligentDriverModel, ds::Float64)
    v = vel(veh.state)
    d = v^2 + 2*idm.a_max*ds
    d < 0 && return 0 # We have already passed the point we are trying to get to
    vf = min(idm.v_des, sqrt(d))
    2*ds/(vf + v)
end

# Compute the time it takes to travel a given distance at the current vehicle velocity
time_to_cross_distance_const_vel(veh::BlinkerVehicle, ds::Float64) = ds / vel(veh.state)

# Computes the distance between the vehicle and the provided point along the roadway
# If the pt is not on the roadway then it is the distance to the projection onto the road
function distance_to_point(veh::BlinkerVehicle, roadway::Roadway, pt::VecSE2)
    s0 = posf(veh.state).s
    lane = get_lane(roadway, veh.state.veh_state)
    s_end = Frenet(pt, lane, roadway).s
    s_end - s0
end

function min_dist(scene, egoid)
    minval = 10000
    (length(scene) == 0 || !has_veh(egoid, scene)) && return minval
    ego_pos = posg(get_by_id(scene, egoid).state)
    for (i,veh) in enumerate(scene)
        if veh.id != egoid
            pos = posg(veh.state)
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
    return false
end

# Check if a given car has reached the end of the roadway
function end_of_road(veh, roadway)
    veh = Vehicle(veh)
    s = posf(veh.state).s
    lane = get_lane(roadway, veh)
    s_end = lane.curve[end].s
    return s >= s_end
end

# Removes cars that have reached the end of the lane
function AutomotiveDrivingModels.run_callback(c::CleanSceneCallback, scenes::Vector{BlinkerScene}, actions::Nothing, roadway::R, models::Dict{I,M}, tick::Int) where {F,I,R,M<:DriverModel}
    for (i,veh) in enumerate(scenes[tick])
        if end_of_road(veh, roadway)
           deleteat!(scenes[tick], findfirst(veh.id, scenes[tick]))
       end
    end
    return ego_collides(scenes[tick], c.egoid)
end

