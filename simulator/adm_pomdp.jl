using POMDPs

const OBS_PER_VEH = 4
const ACT_PER_VEH = 7
const Atype = Array{LaneFollowingAccelBlinker}

mutable struct AdversarialADM <: POMDP{BlinkerScene, Atype, Array{Float64}}
    num_vehicles # The number of vehicles represented in the state and action spaces
    num_controllable_vehicles # Number of vehicles that will be part of the action space
    models # The models for the simulation
    roadway # The roadway for the simulation
    egoid # The id of the ego vehicle
    initial_scene # Initial scene
    dt # Simulation timestep
    last_observation # Last observation of the vehicle state
    actions # Set of all actions for the pomdp
    action_to_index # Dictionary mapping actions to dictionary
end

function AdversarialADM(models, roadway, egoid, intial_scene, dt)
    num_vehicles = length(intial_scene)
    num_controllable_vehicles = num_vehicles - 1
    actions = Array{Atype}(undef, a_dim(num_controllable_vehicles))
    action_to_index = Dict()
    index = 1
    for ijk in CartesianIndices(Tuple(ACT_PER_VEH for i=1:num_controllable_vehicles))
        a = [index_to_action(ijk.I[i], models[i]) for i in 1:num_controllable_vehicles]
        actions[index] = a
        action_to_index[a] = index
        index += 1
    end
    AdversarialADM(num_vehicles, num_controllable_vehicles, models, roadway, egoid, intial_scene, dt, zeros(num_vehicles*OBS_PER_VEH), actions, action_to_index)
end

o_dim(pomdp::AdversarialADM) = pomdp.num_vehicles*OBS_PER_VEH
a_dim(num_controllable_vehicles::Int) = ACT_PER_VEH^num_controllable_vehicles
a_dim(pomdp::AdversarialADM) = a_dim(pomdp.num_controllable_vehicles)


function index_to_action(action::Int, model)
    das = support(model.da_dist)
    action == 0 && return LaneFollowingAccelBlinker(0, 0, false, false)
    action == 1 && return LaneFollowingAccelBlinker(0, das[1], false, false)
    action == 2 && return LaneFollowingAccelBlinker(0, das[2], false, false)
    action == 3 && return LaneFollowingAccelBlinker(0, das[3], false, false)
    action == 4 && return LaneFollowingAccelBlinker(0, das[4], false, false)
    action == 5 && return LaneFollowingAccelBlinker(0, das[5], false, false)
    action == 6 && return LaneFollowingAccelBlinker(0, 0., true, false)
    action == 7 && return LaneFollowingAccelBlinker(0, 0., false, true)
end

function action_to_string(action::Int)
    action == 0 && return "No disturbance"
    action == 1 && return "hard brake"
    action == 2 && return "soft brake"
    action == 3 && return "do nothing"
    action == 4 && return "soft acc"
    action == 5 && return "hard acc"
    action == 6 && return "toggle goal"
    action == 7 && return "toggle blinker"
end


POMDPs.actions(pomdp::AdversarialADM) = pomdp.actions
POMDPs.actions(pomdp::AdversarialADM, state::Tuple{BlinkerScene, Float64}) = actions(pomdp)
POMDPs.actionindex(pomdp::AdversarialADM, a::Atype) = pomdp.action_to_index[a]

action_probability(pomdp::AdversarialADM, s::BlinkerScene, a::Atype) = prod([exp(action_logprob(pomdp.models[i], a[i])) for i in 1:pomdp.num_controllable_vehicles])

# Converts from vector to state
function POMDPs.convert_s(::Type{BlinkerScene}, s::AbstractArray{Float64}, pomdp::AdversarialADM)
    new_scene = BlinkerScene()
    Nveh = Int(length(s) / OBS_PER_VEH)

    # Loop through the vehicles in the scene, apply action and add to next scene
    for i = 1:Nveh
        j = (i-1)*OBS_PER_VEH + 1
        d = s[j] # Distance along the lane
        v = s[j+1] # velocity
        g = s[j+2] # Goal (lane id)
        b = s[j+3] # blinker

        laneid = Int(g)
        lane = pomdp.roadway[laneid].lanes[1]
        blinker = Bool(b)
        vs = VehicleState(Frenet(lane, d, 0.), pomdp.roadway, v)
        bv = BlinkerVehicle(BlinkerState(vs, blinker, pomdp.models[i].goals[laneid]), VehicleDef(), i)

        if !end_of_road(bv, pomdp.roadway)
            push!(new_scene, bv)
        end
    end
    new_scene
end

# Converts the state of a blinker vehicle to a vector
function to_vec(veh::BlinkerVehicle)
    p = posf(veh.state)
    Float64[p.s,
            vel(veh.state),
            laneid(veh),
            veh.state.blinker]
end


# Convert from state to vector (this one is simple )
function POMDPs.convert_s(::Type{Array{Float64, 1}}, state::BlinkerScene, pomdp::AdversarialADM)
    o = deepcopy(pomdp.last_observation)
    for (ind,veh) in enumerate(state)
        o[(veh.id-1)*OBS_PER_VEH + 1: veh.id*OBS_PER_VEH] .= to_vec(veh)
    end
    pomdp.last_observation = o
    o
end

POMDPs.convert_s(::Type{AbstractArray}, state::BlinkerScene, pomdp::AdversarialADM) = convert_s(Array{Float64,1}, state, pomdp)
POMDPs.convert_s(::Type{AbstractArray}, state::BlinkerScene, pomdp::AdversarialADM) = convert_s(Array{Float64,1}, state, pomdp)

# Returns the intial state of the pomdp simulator
POMDPs.initialstate(pomdp::AdversarialADM, rng::AbstractRNG = Random.GLOBAL_RNG) = pomdp.initial_scene

# Get the reward from the actions taken and the next state
POMDPs.reward(pomdp::AdversarialADM, s::BlinkerScene, a::Atype, sp::BlinkerScene) = iscollision(pomdp, sp)

# Step the scene forward by one timestep and return the next state
function step_scene(pomdp::AdversarialADM, s::BlinkerScene, actions::Atype, rng::AbstractRNG = Random.GLOBAL_RNG)
    new_scene = BlinkerScene()

    # Loop through the vehicles in the scene, apply action and add to next scene
    for (i, veh) in enumerate(s)
        model = pomdp.models[veh.id]
        observe!(model, s, pomdp.roadway, veh.id)

        # Set the forced actions of the model
        if model.force_action
            action = actions[veh.id]
            model.da_force = action.da
            model.toggle_goal_force = action.toggle_goal
            model.toggle_blinker_force = action.toggle_blinker
        end

        a = rand(rng, pomdp.models[veh.id])
        vs_p = propagate(veh, a, pomdp.roadway, pomdp.dt)
        bv = BlinkerVehicle(vs_p, veh.def, veh.id)

        if !end_of_road(bv, pomdp.roadway)
            push!(new_scene, bv)
        end
    end

    return new_scene
end

# The generative interface to the POMDP
function POMDPs.gen(pomdp::AdversarialADM, s::BlinkerScene, a::Atype, rng::Random.AbstractRNG = Random.GLOBAL_RNG)
    # Simulate the scene forward one timestep
    # Try to use the existing simulate function
    sp = step_scene(pomdp, s, a, rng)

    # Get the reward
    r = reward(pomdp, s, a, sp)

    # Extract the observations
    o = convert_s(Array{Float64,1}, sp, pomdp)

    # Return
    (sp=sp, o=o, r=r)
end

# Discount factor for the POMDP (Set to 1 because of the finite horizon)
POMDPs.discount(pomdp::AdversarialADM) = 1.

# Check if there is a collision with the ego vehicle in the scene
iscollision(pomdp::AdversarialADM, s::BlinkerScene) = length(s) > 0 && ego_collides(pomdp.egoid, s)

# The simulation is terminal if there is collision with the ego vehicle or if the maximum simulation time has been reached
function POMDPs.isterminal(pomdp::AdversarialADM, s::BlinkerScene)
    length(s) == 0 || iscollision(pomdp, s)
end


### Deal with the actions

# Rollout a policy and return the observations, actions and rewards
function policy_rollout(pomdp::AdversarialADM, policy, s0; save_scenes = false)
    # Setup vectors to store episode information
    Nmax, osz, asz = 300, o_dim(pomdp), 1
    observations = Array{Float64, 2}(undef, Nmax, osz)
    actions = Array{Any}(undef, Nmax)
    rewards = Array{Float64}(undef, Nmax)
    scenes = []

    # Setup initial state and observation
    s, o = s0, convert_s(Vector{Float64}, s0, pomdp)

    i = 0
    while !isterminal(pomdp, s)
        save_scenes && push!(scenes,s)
        i += 1
        observations[i, :] .= o
        a = policy(o)
        actions[i] = a
        s, o, r = gen(pomdp, s, a)
        rewards[i] = r
    end
    save_scenes && push!(scenes, s)
    if !save_scenes
        return view(observations, 1:i, :), view(actions, 1:i), view(rewards, 1:i)
    else
        return view(observations, 1:i, :), view(actions, 1:i), view(rewards, 1:i), scenes
    end
end
