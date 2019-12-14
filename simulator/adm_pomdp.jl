using POMDPs

const OBS_PER_VEH = 4
const ACT_PER_VEH = 7

mutable struct AdversarialADM <: POMDP{BlinkerScene, Int, Array{Float64}}
    num_vehicles # The number of vehicles represented in the state and action spaces
    num_controllable_vehicles # Number of vehicles that will be part of the action space
    models # The models for the simulation
    roadway # The roadway for the simulation
    egoid # The id of the ego vehicle
    initial_scene # Initial scene
    dt # Simulation timestep
    last_observation # Last observation of the vehicle state
end

o_dim(pomdp::AdversarialADM) = pomdp.num_vehicles*OBS_PER_VEH
a_dim(pomdp::AdversarialADM) = ACT_PER_VEH^pomdp.num_controllable_vehicles

function index_to_action(action::Int)
    action == 1 && return LaneFollowingAccelBlinker(0, -0.75, false, false)
    action == 2 && return LaneFollowingAccelBlinker(0, -0.25, false, false)
    action == 3 && return LaneFollowingAccelBlinker(0, 0., false, false)
    action == 4 && return LaneFollowingAccelBlinker(0, 0.25, false, false)
    action == 5 && return LaneFollowingAccelBlinker(0, 0.75, false, false)
    action == 6 && return LaneFollowingAccelBlinker(0, 0., true, false)
    action == 7 && return LaneFollowingAccelBlinker(0, 0., false, true)
end

function action_to_string(action::Int)
    action == 1 && return "hard brake"
    action == 2 && return "soft brake"
    action == 3 && return "do nothing"
    action == 4 && return "soft acc"
    action == 5 && return "hard acc"
    action == 6 && return "toggle goal"
    action == 7 && return "toggle blinker"
end

# Converts the array of actions to LaneFollowingAccelBlinker actions per vehicle
#TODO: Fix for multiple actors
function to_actions(pomdp::AdversarialADM, action::Int)
    actions = fill(LaneFollowingAccelBlinker(0.,0.,false,false), pomdp.num_vehicles)
    @assert pomdp.num_controllable_vehicles == 1
    for i in 1:pomdp.num_controllable_vehicles
        actions[i] = index_to_action(action)
    end
    actions
end

# TODO: Fix for multiple actors
POMDPs.actions(pomdp::AdversarialADM, state::Tuple{BlinkerScene, Float64}) = [1:7 ...]
POMDPs.actions(pomdp::AdversarialADM) = [1:7 ...]

POMDPs.actionindex(pomdp::AdversarialADM, a::Int) = a

# Converts the state of a blinker vehicle to a vector
function to_vec(veh::BlinkerVehicle)
    p = posf(veh.state)
    Float64[p.s,
            vel(veh.state),
            laneid(veh),
            veh.state.blinker]
end

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
        bv = BlinkerVehicle(BlinkerState(vs, blinker, pomdp.models[i].goals[i]), VehicleDef(), i)

        if !end_of_road(bv, pomdp.roadway)
            push!(new_scene, bv)
        end
    end
    new_scene
end

# Convert from state to vector (this one is simple )
function POMDPs.convert_s(::Type{Vector{Float64}}, state::BlinkerScene, pomdp::AdversarialADM)
    o = deepcopy(pomdp.last_observation)
    for (ind,veh) in enumerate(state)
        o[(veh.id-1)*OBS_PER_VEH + 1: veh.id*OBS_PER_VEH] .= to_vec(veh)
    end
    pomdp.last_observation = o
    o
end

# function to_vec(actions::Vector{LaneFollowingAccelBlinker})
#     res = zeros(ACT_PER_VEH*length(actions))
#     for i=1:length(actions)
#         j = (i-1)*ACT_PER_VEH + 1
#         res[j] = actions[i].da
#         res[j+1] = actions[i].toggle_goal ? 1. : -1.
#         res[j+2] = actions[i].toggle_blinker ? 1. : -1.
#     end
#     res
# end

# Get the vector of observations from the state
# function observe_state(pomdp::AdversarialADM, s::BlinkerScene)
#     o = deepcopy(pomdp.last_observation)
#     for (ind,veh) in enumerate(s)
#         o[(veh.id-1)*OBS_PER_VEH + 1: veh.id*OBS_PER_VEH] .= to_vec(veh)
#     end
#     pomdp.last_observation = o
#     o
# end

# Returns the intial state of the pomdp simulator
POMDPs.initialstate(pomdp::AdversarialADM, rng::AbstractRNG = Random.GLOBAL_RNG) = pomdp.initial_scene

# Get the reward from the actions taken and the next state
function reward(pomdp::AdversarialADM, a::Array{LaneFollowingAccelBlinker}, sp::BlinkerScene)
    isterm = isterminal(pomdp, sp)
    iscol = iscollision(pomdp, sp)

    reward = 0
    for (ind,veh) in enumerate(sp)
        i = veh.id
        if i != pomdp.egoid
            reward += get_actions_logpd(pomdp.models[i], a[i])
        end
    end
    if length(sp) > 0
        reward = reward / length(sp)
        # reward -= 0.1*min_dist(sp, pomdp.egoid)
    end
    if reward > 0
        error("what? ")
    end

    if isterm && !iscol
        reward += -10000
    end
    reward
end

function step_scene(pomdp::AdversarialADM, s::BlinkerScene, actions::Array{LaneFollowingAccelBlinker}, rng::AbstractRNG)
    new_scene = BlinkerScene()

    # Loop through the vehicles in the scene, apply action and add to next scene
    for (i, veh) in enumerate(s)
        observe!(pomdp.models[veh.id], s, pomdp.roadway, veh.id)

        # Set the forced actions of the model
        action = actions[veh.id]
        pomdp.models[veh.id].da_force = action.da
        pomdp.models[veh.id].toggle_goal_force = action.toggle_goal
        pomdp.models[veh.id].toggle_blinker_force = action.toggle_blinker

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
function POMDPs.gen(pomdp::AdversarialADM, s::BlinkerScene, a::Int, rng::Random.AbstractRNG = Random.GLOBAL_RNG)
    # Extract the actions that are going to be used
    actions = to_actions(pomdp, a)

    # Simulate the scene forward one timestep
    # Try to use the existing simulate function
    sp = step_scene(pomdp, s, actions, rng)

    # Get the reward
    r = reward(pomdp, actions, sp)

    # Extract the observations
    o = convert_s(Vector{Float64}, sp, pomdp)

    # Return
    (sp=sp, o=o, r=r)
end

# Discount factor for the POMDP (Set to 1 because of the finite horizon)
POMDPs.discount(pomdp::AdversarialADM) = 0.95

# Check if there is a collision with the ego vehicle in the scene
iscollision(pomdp::AdversarialADM, s::BlinkerScene) = length(s) > 0 && ego_collides(pomdp.egoid, s)

# The simulation is terminal if there is collision with the ego vehicle or if the maximum simulation time has been reached
function POMDPs.isterminal(pomdp::AdversarialADM, s::BlinkerScene)
    length(s) == 0 || iscollision(pomdp, s)
end


### Deal with the actions

# Compiles a set of nominal actions for each vehicle in the scene
# Number of vehicles is included so that the action space can stay the same size
function nominal_action(pomdp::AdversarialADM, s::BlinkerScene)
    actions = Array{LaneFollowingAccelBlinker}(undef, pomdp.num_vehicles)
    for (i,veh) in enumerate(s)
        actions[veh.id] = LaneFollowingAccelBlinker(0., 0., false, false)
    end
    to_vec(actions)
end

# compiles a set of random actions
function random_action(pomdp::AdversarialADM, s::BlinkerScene, rng::AbstractRNG = Random.GLOBAL_RNG)
    actions = Array{LaneFollowingAccelBlinker}(undef, pomdp.num_controllable_vehicles)
    for (i,veh) in enumerate(s)
        if pomdp.models[veh.id].force_action
            actions[veh.id] = random_action(pomdp.models[veh.id], rng)
        end
    end
    to_vec(actions)
end

# Rollout a policy and return the observations, actions and rewards
function policy_rollout(pomdp::AdversarialADM, policy, s0; save_scenes = false)
    # Setup vectors to store episode information
    Nmax, osz, asz = 300, o_dim(pomdp), 1
    observations = Array{Float64, 2}(undef, Nmax, osz)
    actions = Array{Float64}(undef, Nmax)
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

function mcts_rollout(pomdp::AdversarialADM, s, depth = 0, rng::AbstractRNG = Random.GLOBAL_RNG)
    tot_r = 0
    mul = 1
    while !isterminal(pomdp, s)
        actions = random_action(pomdp, s, rng)
        s, o, r = gen(pomdp, s, actions)
        tot_r += r*mul
        mul *= discount(pomdp)
    end
    tot_r
end


