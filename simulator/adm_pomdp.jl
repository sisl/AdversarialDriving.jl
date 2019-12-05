using POMDPs

const OBS_PER_VEH = 6
const ACT_PER_VEH = 3

mutable struct AdversarialADM <: POMDP{Tuple{BlinkerScene, Float64}, Array{Float64}, Array{Float64}}
    num_vehicles # The number of vehicles represented in the state and action spaces
    models # The models for the simulation
    roadway # The roadway for the simulation
    egoid # The id of the ego vehicle
    dt # timestep of the simulation
    T # Max time of the simulation
    initial_scene # Initial scene
end

o_dim(pomdp::AdversarialADM) = pomdp.num_vehicles*OBS_PER_VEH
a_dim(pomdp::AdversarialADM) = pomdp.num_vehicles*ACT_PER_VEH
max_steps(pomdp::AdversarialADM) = Int64(round(pomdp.T / pomdp.dt, RoundUp)) + 1
dt(pomdp::AdversarialADM) = pomdp.dt

# Get the scene from the POMDP state
get_scene(s::Tuple{BlinkerScene, Float64}) = s[1]

# Get the simulation time from the POMDP state
get_t(s::Tuple{BlinkerScene, Float64}) = s[2]

# Converts the state of a blinker vehicle to a vector
function to_vec(veh::BlinkerVehicle)
    p = posg(veh.state)
    Float64[p.x, p.y, p.θ, vel(veh.state), laneid(veh), veh.state.blinker]
end

# Converts the array of actions to LaneFollowingAccelBlinker actions per vehicle
function to_actions(pomdp::AdversarialADM, action_vec::Array{Float64})
    actions = fill(LaneFollowingAccelBlinker(0.,0.,false,false), pomdp.num_vehicles)
    for i in 1:pomdp.num_vehicles
        j = (i-1)*ACT_PER_VEH + 1
        da = action_vec[j]
        toggle_goal = action_vec[j+1] > 0.
        toggle_blinker = action_vec[j+2] > 0.
        # Note that the acceleration is set to 0 and will not be used
        actions[i] = LaneFollowingAccelBlinker(0, da, toggle_goal, toggle_blinker)
    end
    actions
end

function to_vec(actions::Vector{LaneFollowingAccelBlinker})
    res = zeros(ACT_PER_VEH*length(actions))
    for i=1:length(actions)
        j = (i-1)*ACT_PER_VEH + 1
        res[j] = actions[i].da
        res[j+1] = actions[i].toggle_goal ? 1. : -1.
        res[j+2] = actions[i].toggle_blinker ? 1. : -1.
    end
    res
end

# Get the vector of observations from the state
function observe_state(pomdp::AdversarialADM, s::Tuple{BlinkerScene, Float64})
    o = zeros(pomdp.num_vehicles*OBS_PER_VEH)
    for (ind,veh) in enumerate(get_scene(s))
        o[(veh.id-1)*OBS_PER_VEH + 1: veh.id*OBS_PER_VEH] .= to_vec(veh)
    end
    o
end

# Returns the intial state of the pomdp simulator
function POMDPs.initialstate(pomdp::AdversarialADM, rng::AbstractRNG = Random.GLOBAL_RNG)
    return (pomdp.initial_scene, 0.)
end

# Get the reward from the actions taken and the next state
function reward(pomdp::AdversarialADM, a::Array{LaneFollowingAccelBlinker}, sp::Tuple{BlinkerScene, Float64})
    isterm = isterminal(pomdp, sp)
    iscol = iscollision(pomdp, sp)

    reward = 0
    for (ind,veh) in enumerate(get_scene(sp))
        i = veh.id
        if i != pomdp.egoid
            reward += get_actions_logpd(pomdp.models[i], a[i])
        end
    end
    if length(get_scene(sp)) > 0
        reward = reward / length(get_scene(sp))
        reward -= 0.1*min_dist(get_scene(sp), pomdp.egoid)
    end

    if isterm && !iscol
        reward = -10000
    end
    # if iscol
    #     println("found a collision!")
    # end
    reward
end

function step_scene(pomdp::AdversarialADM, s::Tuple{BlinkerScene, Float64}, actions::Array{LaneFollowingAccelBlinker}, rng::AbstractRNG)
    new_scene = BlinkerScene()
    next_t = get_t(s) + pomdp.dt

    # Loop through the vehicles in the scene, apply action and add to next scene
    for (i, veh) in enumerate(get_scene(s))
        observe!(pomdp.models[veh.id], get_scene(s), pomdp.roadway, veh.id)

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

    return (new_scene, next_t)
end

# The generative interface to the POMDP
function POMDPs.gen(pomdp::AdversarialADM, s::Tuple{BlinkerScene, Float64}, a::Array{Float64}, rng::Random.AbstractRNG = Random.GLOBAL_RNG)
    # Extract the actions that are going to be used
    actions = to_actions(pomdp, a)

    # Simulate the scene forward one timestep
    # Try to use the existing simulate function
    sp = step_scene(pomdp, s, actions, rng)

    # Get the reward
    r = reward(pomdp, actions, sp)

    # Extract the observations
    o = observe_state(pomdp, sp)

    # Return
    (sp=sp, o=o, r=r)
end

# Discount factor for the POMDP (Set to 1 because of the finite horizon)
POMDPs.discount(pomdp::AdversarialADM) = 1.

# Check if there is a collision with the ego vehicle in the scene
iscollision(pomdp::AdversarialADM, s::Tuple{BlinkerScene, Float64}) = length(get_scene(s)) > 0 && ego_collides(pomdp.egoid, get_scene(s))

# The simulation is terminal if there is collision with the ego vehicle or if the maximum simulation time has been reached
function POMDPs.isterminal(pomdp::AdversarialADM, s::Tuple{BlinkerScene, Float64})
    length(get_scene(s)) == 0 || get_t(s) >= pomdp.T || iscollision(pomdp, s)
end


### Deal with the actions

# Compiles a set of nominal actions for each vehicle in the scene
# Number of vehicles is included so that the action space can stay the same size
function nominal_action(pomdp::AdversarialADM, s::Tuple{BlinkerScene, Float64})
    actions = Array{LaneFollowingAccelBlinker}(undef, pomdp.num_vehicles)
    for (i,veh) in enumerate(get_scene(s))
        actions[veh.id] = LaneFollowingAccelBlinker(0., 0., false, false)
    end
    to_vec(actions)
end

# compiles a set of random actions
function random_action(pomdp::AdversarialADM, s::Tuple{BlinkerScene, Float64}, rng::AbstractRNG = Random.GLOBAL_RNG)
    actions = Array{LaneFollowingAccelBlinker}(undef, pomdp.num_vehicles)
    for (i,veh) in enumerate(get_scene(s))
        actions[veh.id] = random_action(pomdp.models[veh.id], rng)
    end
    to_vec(actions)
end

# Rollout a policy and return the observations, actions and rewards
function policy_rollout(pomdp::AdversarialADM, policy, s0; save_scenes = false)
    # Setup vectors to store episode information
    Nmax, osz, asz = max_steps(pomdp), o_dim(pomdp), a_dim(pomdp)
    observations = Array{Float64, 2}(undef, Nmax, osz)
    actions = Array{Float64, 2}(undef, Nmax, asz)
    rewards = Array{Float64}(undef, Nmax)
    scenes = []

    # Setup initial state and observation
    s, o = s0, observe_state(pomdp, s0)

    i = 0
    while !isterminal(pomdp, s)
        save_scenes && push!(scenes, get_scene(s))
        i += 1
        observations[i, :] .= o
        a = policy(o)
        actions[i, :] .= a
        s, o, r = gen(pomdp, s, a)
        rewards[i] = r
    end
    save_scenes && push!(scenes, get_scene(s))
    if !save_scenes
        return view(observations, 1:i, :), view(actions, 1:i, :), view(rewards, 1:i)
    else
        return view(observations, 1:i, :), view(actions, 1:i, :), view(rewards, 1:i), scenes
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


