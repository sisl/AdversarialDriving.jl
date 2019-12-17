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
    veh_state::VehicleState # position and velocity
    blinker::Bool # Whether or not the blinker is on
    goals::Vector{Int} # The list of possible goals that this vehicle can have
end

AutomotiveDrivingModels.posf(s::BlinkerState) = posf(s.veh_state)
AutomotiveDrivingModels.posg(s::BlinkerState) = s.veh_state.posG
vel(s::BlinkerState) = s.veh_state.v

# Define a new Vehicle with the BlinkerVehicle
BlinkerVehicle = Entity{BlinkerState, VehicleDef, Int64}

# Constructor for making a blinker vehicle
function BV(posG::VecSE2, v::Float64, goals::Vector{Int}, goal_lane::Int, blinker, id, roadway::Roadway)
    vs = VehicleState(posG, roadway, v)
    bv = BlinkerVehicle(BlinkerState(vs, blinker, goals), VehicleDef(), id)
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
    toggle_goal::Bool
    toggle_blinker::Bool
end

# The function that propogates the new action
function AutomotiveDrivingModels.propagate(veh::BlinkerVehicle, action::LaneFollowingAccelBlinker, roadway::Roadway, Δt::Float64)
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
    ds = vel(veh.state)
    Δs = max(0,ds*Δt + 0.5*a*Δt*Δt)
    v₂ = max(0,ds + a*Δt)

    roadind = move_along(posf(veh.state).roadind, roadway, Δs)
    posG = roadway[roadind].pos
    posF = Frenet(roadind, roadway, t=posf(veh.state).t, ϕ=posf(veh.state).ϕ)

    # Set the blinker state and return
    new_blink = action.toggle_blinker ? !veh.state.blinker : veh.state.blinker
    BlinkerState(VehicleState(posG, posF, v₂), new_blink, veh.state.goals)
end


# Definition of the T-Intersection DriverModel with observe! and rand()
############################################################################
############################################################################

# Define a driving model for a T-intersection IDM model
@with_kw mutable struct TIDM <: DriverModel{LaneFollowingAccelBlinker}
    idm::IntelligentDriverModel = IntelligentDriverModel()

    # Defines the stochastic actions of the agents
    ttc_threshold = 5 # threshold through intersection
    da_dist::Normal = Normal(0,1)# Distributions over acc
    toggle_goal_dist::Bernoulli = Bernoulli(1e-14) # Distribution over changing goals
    toggle_blinker_dist::Bernoulli = Bernoulli(1e-14) # Distribution over toggling signal

    # The members below here are for control of AST
    force_action::Bool = false # Whether or not we should use the stored actions
    da_force::Float64 = 0. # The acceleration to apply
    toggle_goal_force::Bool = false # Whether to toggle a goal
    toggle_blinker_force::Bool = false # Whether to toggle the turn signal

    # Describes the intersection and rules of the road
    yields_way::Dict{Int64, Vector{Int64}} = Dict() # Lane priorities
    intersection_enter_loc::Dict{Int64, VecSE2} = Dict() # Entry location of intersection
    intersection_exit_loc::Dict{Int64, VecSE2} = Dict()  # Exit location of intersection
    goals::Dict{Int64, Vector{Int64}} = Dict() # Possible goals of each lane
    should_blink::Dict{Int64, Bool} = Dict()  # Wether or not the blinker should be on
end

# Easy generation function for getting a driving that is controllable by AST
function generate_TIDM_AST(yields_way, intersection_enter_loc, intersection_exit_loc, goals, should_blink)
    TIDM(   force_action = true,
            yields_way = yields_way,
            intersection_enter_loc = intersection_enter_loc,
            intersection_exit_loc = intersection_exit_loc,
            goals = goals,
            should_blink = should_blink,
            )
end

# Make a copy of an existing model, while replacing the id, and probabilities
function generate_TIDM_AST(template::TIDM; p_toggle_blinker, p_toggle_goal, σ2a)
    TIDM(
            da_dist = Normal(0,σ2a),
            toggle_goal_dist = Bernoulli(p_toggle_goal),
            toggle_blinker_dist = Bernoulli(p_toggle_blinker),
            force_action = template.force_action,
            yields_way = template.yields_way,
            intersection_enter_loc = template.intersection_enter_loc,
            intersection_exit_loc = template.intersection_exit_loc,
            goals = template.goals,
            should_blink = template.should_blink,
            )
end

# Get the probability density of the specified action
function get_actions_logpd(model::TIDM, action::LaneFollowingAccelBlinker)
    a_pd = logpdf(model.da_dist, action.da)
    goal_pm = logpdf(model.toggle_goal_dist, action.toggle_goal)
    blinker_pm = logpdf(model.toggle_blinker_dist, action.toggle_blinker)
    tot = a_pd + goal_pm + blinker_pm
    @assert isfinite(tot)
    tot
end

# Gets a random action from the model, ignoring the force flag
function random_action(model::TIDM, rng::AbstractRNG = Random.GLOBAL_RNG)
    da = rand(rng, model.da_dist)
    toggle_goal = rand(rng, model.toggle_goal_dist)
    toggle_blinker = rand(rng, model.toggle_blinker_dist)
    LaneFollowingAccelBlinker(model.idm.a, da, toggle_goal, toggle_blinker)
end

# Sample an action from TIDM model
function Base.rand(rng::AbstractRNG, model::TIDM)
    if model.force_action # Use the forced actions
        LaneFollowingAccelBlinker(model.idm.a, model.da_force, model.toggle_goal_force, model.toggle_blinker_force)
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

    blinker_match = [veh.state.blinker == model.should_blink[l] for l in possible_lanes]
    if findfirst(blinker_match) != nothing
        return possible_lanes[findfirst(blinker_match)]
    else
        return possible_lanes[1]
    end
end

# Observe function for TIDM
function AutomotiveDrivingModels.observe!(model::TIDM, scene::BlinkerScene, roadway::Roadway, egoid::Int64)
    # Pull out the necessary quantities
    vehicle_index = findfirst(egoid, scene)
    ego = scene[vehicle_index]
    li = laneid(ego)

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
    BlinkerVehicle(BlinkerState(VehicleState(posF, roadway, vel(veh.state)), veh.state.blinker, veh.state.goals), veh.def, veh.id)
end

# Returns a new vehicle that is a copy of the provided vehicle except with the specified id
set_veh_id(veh::BlinkerVehicle, id::Int) = BlinkerVehicle(veh.state, veh.def, id)

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

