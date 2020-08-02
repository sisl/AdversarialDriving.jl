## Definition of new vehicle state, vehicle, scene, and action

# Struct that contains noise for position and velocity (assumed to be in s, t coords)
@with_kw struct Noise
    pos::VecE2 = VecE2(0,0)
    vel::Float64 = 0
end

abstract type NoisyState end

# State for a pedestrian that has noise
@with_kw struct NoisyPedState <: NoisyState
    veh_state::VehicleState # Noise on position and velocity
    noise::Noise # Noise on position and velocity
end

# State for vehicles with blinkers, lane intention and noise
@with_kw struct BlinkerState <: NoisyState
    veh_state::VehicleState # position and velocity
    blinker::Bool # Whether or not the blinker is on
    goals::Array{Int} # The list of possible goals that this vehicle can have
    noise::Noise # Noise on position and velocity
end

# Creates a copy of the state  with the new specified vehicle state
update_veh_state(s::NoisyPedState, veh_state::VehicleState) = NoisyPedState(veh_state, s.noise)
update_veh_state(s::BlinkerState, veh_state::VehicleState) = BlinkerState(veh_state, s.blinker, s.goals, s.noise)

# Position and velocity functions
AutomotiveSimulator.posf(s::NoisyState) = posf(s.veh_state)
AutomotiveSimulator.posg(s::NoisyState) = posg(s.veh_state)
AutomotiveSimulator.vel(s::NoisyState)  = vel(s.veh_state)
AutomotiveSimulator.velf(s::NoisyState) = velf(s.veh_state)
AutomotiveSimulator.velg(s::NoisyState) = velg(s.veh_state)

# Get the blinker of a state or entity
blinker(s) = false # default behavior
blinker(s::BlinkerState) = s.blinker
blinker(veh::Entity) = blinker(veh.state)

# Get the goals of a state or entity
goals(s) = [laneid(s)] # default behavior
goals(s::BlinkerState) = s.goals
goals(veh::Entity) = goals(veh.state)

# Get the noise of a state or entity
noise(veh) = Noise() # default behavior
noise(s::NoisyState) = s.noise
noise(veh::Entity) = noise(veh.state)


# Makes a copy of the scene with the noise added to the vehicles in the state
function noisy_scene(scene::Scene, roadway::Roadway)
    noisy_scene = Scene(Entity)
    for (i,veh) in enumerate(scene)
        push!(noisy_scene, noisy_entity(veh, roadway))
    end
    noisy_scene
end

# Makes a copy of the entity but with the noise applied to the VehicleState
function noisy_entity(ent, roadway::Roadway)
    f = posf(ent)
    Δs, Δt = noise(ent).pos
    noisy_f = Frenet(get_lane(roadway, ent), f.s + Δs, f.t + Δt, f.ϕ)
    noisy_g = posg(noisy_f, roadway)
    noisy_v = vel(ent) + noise(ent).vel
    noisy_vs = VehicleState(noisy_g, noisy_f, noisy_v)
    Entity(update_veh_state(ent.state, noisy_vs), ent.def, ent.id)
end

const PEDESTRIAN_DEF = VehicleDef(AgentClass.PEDESTRIAN, 1.0, 1.0)

# Constructor for making a blinker vehicle
function BlinkerVehicle(;roadway::Roadway, lane::Int, s::Float64, v::Float64, id::Int, goals::Array{Int}, blinker::Bool, t::Float64 = 0., ϕ::Float64 = 0., noise::Noise = Noise())
    f = Frenet(roadway, lane, s, t, ϕ)
    bs = BlinkerState(VehicleState(f, roadway, v), blinker, goals, noise)
    Entity(bs, VehicleDef(), id)
end

# Constructor for making a noisy pedestrian
function NoisyPedestrian(;roadway::Roadway, lane::Int, s::Float64, v::Float64, id::Int, t::Float64=0., ϕ::Float64 = 0., noise::Noise = Noise())
    f = Frenet(roadway_lane(roadway, lane), s, t, ϕ)
    ps = NoisyPedState(VehicleState(f, roadway, v), noise)
    Entity(ps, PEDESTRIAN_DEF, id)
end

# Instructions for rendering the Blinker Vehicle
function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, veh::Entity{BlinkerState, VehicleDef, Int64})
    reg_veh = Entity(veh.state.veh_state, veh.def, veh.id)
    add_renderable!(rendermodel, FancyCar(car=reg_veh))
    #TODO: Add noisy ghost
    li = laneid(veh)
    bo = BlinkerOverlay(on = blinker(veh), veh = reg_veh, right=Tint_signal_right[li])
    add_renderable!(rendermodel, bo)
    return rendermodel
end

# Instructions for rendering the noisy pedestrian
function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, ped::Entity{NoisyPedState, VehicleDef, Int64})
    reg_ped = Entity(ped.state.veh_state, ped.def, ped.id)
    add_renderable!(rendermodel, FancyPedestrian(ped=reg_ped))
    #TODO: Add noisy ghost
    return rendermodel
end


## Definition of BlinkerVehicleControl action with helpers and propagate
abstract type Disturbance end

# Define a new action that conrols a blinker vehicle
@with_kw struct BlinkerVehicleControl <: Disturbance
    a::Float64 = 0 # Acceleration along lane (set by the model)
    da::Float64 = 0 # Acceleration disturbance along lane
    toggle_goal::Bool = false # Switches vehicle turn intention
    toggle_blinker::Bool = false # Toggles the turn signal
    noise::Noise = Noise() #
end

# Define a new action that controls a pedestrian
@with_kw struct PedestrianControl <: Disturbance
    a::VecE2 = VecE2(0., 0.)
    da::VecE2 = VecE2(0., 0.)
    noise::Noise = Noise()
end

# The function that propagates the PedestrianControl action
function AutomotiveSimulator.propagate(ped::Entity{NoisyPedState, D, I}, action::PedestrianControl, roadway::Roadway, Δt::Float64) where {D, I}
    starting_lane = laneid(ped)
    vs_entity = Entity(ped.state.veh_state, ped.def, ped.id)
    a_lat_lon = reverse(action.a + action.da)
    vs = propagate(vs_entity, LatLonAccel(a_lat_lon...), roadway, Δt)
    vs = VehicleState(vs.posG, vs.posF, clamp(vs.v, -3, 3)) # Max pedestrian speed
    nps = NoisyPedState(set_lane(vs, laneid(ped), roadway), action.noise)
    @assert starting_lane == laneid(nps)
    nps
end

# The function that propogates the BlinkerVehicleControl action
function AutomotiveSimulator.propagate(veh::Entity{BlinkerState, D, I}, action::BlinkerVehicleControl, roadway::Roadway, Δt::Float64) where {D,I}
    # set the new goal
    vs = veh.state.veh_state
    if action.toggle_goal
        gs = goals(veh)
        curr_index = findfirst(gs .== laneid(veh))
        @assert !isnothing(curr_index)
        new_goal = gs[curr_index % length(gs) + 1]
        if can_have_goal(veh, new_goal, roadway)
            vs = set_lane(vs, new_goal, roadway)
        end
    end

    starting_lane = laneid(vs)

    # Update the kinematics of the vehicle (don't allow v < 0)
    vs_entity = Entity(vs, veh.def, veh.id)
    vs = propagate(vs_entity, LaneFollowingAccel(action.a + action.da), roadway, Δt)

    # Set the blinker state and return
    new_blink = action.toggle_blinker ? !blinker(veh) : blinker(veh)
    bs = BlinkerState(vs, new_blink, goals(veh), action.noise)
    @assert starting_lane == laneid(bs)
    bs
end

## Define a pedestrian control module

# Define the wrapper for the adversarial pedestrian
@with_kw mutable struct AdversarialPedestrian <: DriverModel{PedestrianControl}
    idm::IntelligentDriverModel = IntelligentDriverModel(v_des= 1.0)
    next_action::PedestrianControl = PedestrianControl()
    ignore_idm = false
end

# Sample an action from AdversarialPedestrian model
function Base.rand(rng::AbstractRNG, model::AdversarialPedestrian)
    na = model.next_action
    if !model.ignore_idm
        return PedestrianControl((model.idm.a, 0), na.da, na.noise)
    else
        return na
    end
end

# Observe function for AdversarialPedestrian model
function AutomotiveSimulator.observe!(model::AdversarialPedestrian, scene::Scene, roadway::Roadway, egoid::Int64)
    ego = get_by_id(scene, egoid)
    ego_v = vel(ego)
    track_longitudinal!(model.idm, ego_v, NaN, NaN)
    model
end

## Definition of Blindspot
struct Blindspot
    θ::Float64 # Offset of the blindspot from vehicle direction
    ϕ::Float64 # Width of the blindspot
end

function in_blindspot(egopos::VecSE2{Float64}, blindspot::Blindspot, otherpos::VecSE2{Float64})
    Δy = otherpos.y - egopos.y
    Δx = otherpos.x - egopos.x
    ψ = atan(Δy, Δx)
    θ = blindspot.θ + egopos.θ
    (ψ > θ - blindspot.ϕ / 2.) && (ψ < θ  + blindspot.ϕ / 2.)
end

struct RenderableBlindspot
    pos::VecSE2
    blindspot::Blindspot
    length::Float64
    color::Colorant
end

function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, rb::RenderableBlindspot)
    θ = rb.blindspot.θ + rb.pos.θ
    low = θ - rb.blindspot.ϕ / 2.
    high = θ + rb.blindspot.ϕ / 2.
    pt1 = VecE2{Float64}(rb.length*cos(low), rb.length*sin(low))
    pt2 = VecE2{Float64}(rb.length*cos(high), rb.length*sin(high))
    p = VecE2(rb.pos)
    pts = [p, p+pt1, p+pt2, p]

    add_instruction!(
        rendermodel, AutomotiveVisualization.render_closed_line,
        (pts,  rb.color, 0.1, rb.color),
        coordinate_system=:scene
    )
    return rendermodel
end

## Definition of the T-Intersection DriverModel with observe! and rand()

# Define a driving model for a T-intersection IDM model
@with_kw mutable struct TIDM <: DriverModel{BlinkerVehicleControl}
    idm::IntelligentDriverModel = IntelligentDriverModel() # underlying idm
    noisy_observations::Bool = false # Whether or not this model gets noisy observations
    ttc_threshold = 7 # threshold through intersection
    next_action::BlinkerVehicleControl = BlinkerVehicleControl() # The next action that the model will do (for controllable vehicles)

    # Describes the intersection and rules of the road
    yields_way::Dict{Int64, Array{Int64}} = Dict() # Lane priorities
    intersection_enter_loc::Dict{Int64, VecSE2} = Dict() # Entry location of intersection
    intersection_exit_loc::Dict{Int64, VecSE2} = Dict()  # Exit location of intersection
    goals::Dict{Int64, Array{Int64}} = Dict() # Possible goals of each lane
    should_blink::Dict{Int64, Bool} = Dict()  # Wether or not the blinker should be on

    blindspot::Union{Blindspot, Nothing} = nothing # Blindspot of the vehicle
end

# Sample an action from TIDM model
function Base.rand(rng::AbstractRNG, model::TIDM)
    na = model.next_action
    BlinkerVehicleControl(model.idm.a, na.da, na.toggle_goal, na.toggle_blinker, na.noise)
 end

# Observe function for TIDM
function AutomotiveSimulator.observe!(model::TIDM, input_scene::Scene, roadway::Roadway, egoid::Int64)
    # If this model is susceptible to noisy observations, adjust all the agents by noise
    scene = model.noisy_observations ? noisy_scene(input_scene, roadway) : input_scene

    # Get the ego and the ego lane
    ego = get_by_id(scene, egoid)
    ego_v = vel(ego)
    li = laneid(ego)

    # Compute ego headway to intersection and time to cross
    intrsxn_Δs = distance_to_point(ego, roadway, model.intersection_enter_loc[li])
    intrsxn_exit_Δs = distance_to_point(ego, roadway, model.intersection_exit_loc[li])
    time_to_cross = time_to_cross_distance_const_acc(ego, model.idm, distance_to_point(ego, roadway, model.intersection_exit_loc[li]))

    # Get headway to the forward car
    fore = find_neighbor(scene, roadway, ego, targetpoint_ego = VehicleTargetPointFront(), targetpoint_neighbor = VehicleTargetPointRear())
    fore_v, fore_Δs = isnothing(fore.ind) ? (NaN, Inf) : (vel(scene[fore.ind]), fore.Δs)

    # Check to see if ego car has right of way
    has_right_of_way = true
    lanes_to_yield_to = model.yields_way[li]
    vehicles_to_yield_to = []
    for (i,veh) in enumerate(scene)
        # if the vehicle is the ego vehicle then move on
        veh.id == egoid && continue

        # Check if the other entity is in the blind spot. If so, move one
        !isnothing(model.blindspot) && in_blindspot(posg(ego), model.blindspot, posg(veh)) && continue

        # Check to see if the vehicle is in the ego's lane without the same laneid
        if laneid(ego) != laneid(veh)
            Δs_inlane = compute_inlane_headway(ego, veh, roadway)
            if Δs_inlane < fore_Δs
                fore_Δs = Δs_inlane
                fore_v = vel(veh)
            end
        end

        # If the vehicle is in a lane that the ego should yield to, store it
        if lane_belief(veh, model, roadway) in lanes_to_yield_to
            has_right_of_way = false
            push!(vehicles_to_yield_to, veh)
        end
    end

    # TODO: Reasoning to take place if you enter an intersection
    # if i_headway <-r
        # has_right_of_way = true
    # end

    next_idm = track_longitudinal!(model.idm, ego_v, fore_v, fore_Δs)
    # If the vehicle does not have right of way then stop before the intersection
    if !has_right_of_way
        Nv = length(vehicles_to_yield_to)
        in_intersection = Vector{Bool}(undef, Nv)
        exit_time = Vector{Float64}(undef, Nv)
        enter_time = Vector{Float64}(undef, Nv)
        Δs_in_lane = Vector{Float64}(undef, Nv)
        for (i, veh) in zip(1:Nv, vehicles_to_yield_to)
            _exit = distance_to_point(veh, roadway, model.intersection_exit_loc[laneid(veh)])
            _enter = distance_to_point(veh, roadway, model.intersection_enter_loc[laneid(veh)])
            in_intersection[i] = (_enter < 2 && _exit > 0)
            exit_time[i] = time_to_cross_distance_const_vel(veh, _exit)
            enter_time[i] = time_to_cross_distance_const_vel(veh, _enter)
            Δs_in_lane[i] = compute_inlane_headway(ego, veh, roadway)
        end

        # The intersection is clear of car i if, it exited the intersection in the past, or
        # it will enter the intersection after you have crossed it, or
        # it will have exited a while before you crossed
        intersection_clear = (exit_time .<= 0) .| (enter_time .> time_to_cross) .| (exit_time .+ model.ttc_threshold .< time_to_cross)
        intersection_clear = intersection_clear .& (.!in_intersection)
        intersection_clear = intersection_clear .& (Δs_in_lane .> intrsxn_exit_Δs)
        if !all(intersection_clear)
            # yield to oncoming traffic
            minΔs_to_yield = minimum(Δs_in_lane[.!intersection_clear]) # headways of cars you care about
            # If there isn't a leading car, or if it is past the intersection, use intersection point, otherwise use car
            if isinf(fore_Δs) || minΔs_to_yield < fore_Δs || (intrsxn_Δs > 0 && intrsxn_Δs < fore_Δs)
                next_idm = track_longitudinal!(model.idm, ego_v, 0., min(minΔs_to_yield, intrsxn_Δs))
            end
        end

    end
    model.idm = next_idm
    model
end

# function find_nearest_car_forward(ego, scene, roadway)
#     for (i,veh) in enumerate(scene)
#         if veh.id != egoid &&
# end

function compute_inlane_headway(ego, veh, roadway)
    elane = laneid(ego)
    v = Entity(set_lane(veh.state.veh_state, elane, roadway), veh.def, veh.id)
    f = posf(v)
    Δs = f.s - v.def.length/2. - posf(ego).s - targetpoint_delta(VehicleTargetPointFront(), ego)
    tbound = get_lane(roadway, ego.state.veh_state).width / 2.
    (abs(f.t) <= tbound && Δs > 0) ? Δs : Inf
end

# Get the belief of the lane of the specificed vehicle
function lane_belief(veh::Entity, model::TIDM, roadway::Roadway)
    possible_lanes = model.goals[laneid(veh)]
    length(possible_lanes) == 1 && return possible_lanes[1]

    possible_lanes = possible_lanes[[can_have_goal(veh, l, roadway) for l in possible_lanes]]
    length(possible_lanes) == 1 && return possible_lanes[1]

    blinker_match = findfirst([blinker(veh) == model.should_blink[l] for l in possible_lanes])
    if !isnothing(blinker_match)
        return possible_lanes[blinker_match]
    else
        return model.goals[laneid(veh)][1]
    end
end


## Definition of helper functions for managing vehicles on the road
AutomotiveSimulator.get_lane(roadway::Roadway, state::BlinkerState) = roadway[posf(state).roadind.tag]
AutomotiveSimulator.get_lane(roadway::Roadway, state::NoisyPedState) = roadway[posf(state).roadind.tag]

# Get the laneid of a vehicle
laneid(veh) = posf(veh).roadind.tag.segment

# Returns a vehicle with the same state, but projected onto the desired lane
function set_lane(state::VehicleState, laneid::Int, roadway::Roadway)
    desired_lane = roadway_lane(roadway, laneid)
    posF = Frenet(posg(state), desired_lane, roadway)
    VehicleState(posF, roadway, vel(state))
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
    lane = get_lane(roadway, veh)
    s_end = Frenet(pt, lane, roadway).s
    s_end - s0
end

# computes the closest distance between the ego vehicle and another vehicle on the road
function min_dist(scene, egoid)
    minval = 10000
    (length(scene) == 0 || !(egoid in scene)) && return minval
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

# Check if the ego vehicle is in collision with another car
function ego_collides(egoid, scene)
    if !(egoid in scene)
        return false
    end

    ego = get_by_id(scene, egoid)
    for (i,veh) in enumerate(scene)
        if egoid != veh.id && collision_checker(ego, veh) && vel(ego) > 0.1
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
function end_of_road(veh, roadway, set_end)
    s = posf(veh).s

    lane = get_lane(roadway, veh)
    s_end = lane.curve[end].s
    return s >= min(s_end, set_end)
end

# Removes cars that have reached the end of the lane
function AutomotiveSimulator.run_callback(c::CleanSceneCallback, scenes::Array{Scene}, actions::Nothing, roadway::R, models::Dict{I,M}, tick::Int) where {F,I,R,M<:DriverModel}
    for (i,veh) in enumerate(scenes[tick])
        if end_of_road(veh, roadway)
           deleteat!(scenes[tick], findfirst(veh.id, scenes[tick]))
       end
    end
    return ego_collides(scenes[tick], c.egoid)
end

## This stuff is used when scene has different types
Base.in(id, scene::Scene) = !isnothing(findfirst(id, scene))

function Base.findfirst(id, scene::Scene)
    for entity_index in 1 : scene.n
        scene.entities[entity_index].id == id && return entity_index
    end
    nothing
end

function AutomotiveSimulator.get_by_id(scene::Scene, id)
    entity_index = findfirst(id, scene)
    isnothing(entity_index) && throw(BoundsError(scene, [id]))
    scene[entity_index]
end


## Neural Network driving model
@with_kw mutable struct PolicyModel <: DriverModel{LaneFollowingAccel}
    policy
    state = nothing
end

# Sample an action from TIDM model
function Base.rand(rng::AbstractRNG, model::PolicyModel)
    action(model.policy, model.state)
end

# Observe function for TIDM
function AutomotiveSimulator.observe!(model::PolicyModel, input_scene::Scene, roadway::Roadway, egoid::Int64)
    model.state = input_scene
end

