const OBS_PER_VEH = 4 # Position, velocity, lane, blinker
const OBS_PER_VEH_EXPANDED = 30
const Atype = Array{LaneFollowingAccelBlinker}
const ACTIONS = [ LaneFollowingAccelBlinker(0, 0., false, false),
                        LaneFollowingAccelBlinker(0, -3., false, false),
                        LaneFollowingAccelBlinker(0, -1.5, false, false),
                        LaneFollowingAccelBlinker(0, 1.5, false, false),
                        LaneFollowingAccelBlinker(0, 3., false, false),
                        LaneFollowingAccelBlinker(0, 0., true, false), # toggle goal
                        LaneFollowingAccelBlinker(0, 0., false, true) # toggle blinker
                        ]
const ACTION_PROB = [1 - (4e-3 + 2e-2), 1e-3, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3]

@with_kw mutable struct AdversarialDrivingMDP <: MDP{Scene, Atype}
    num_vehicles # The number of vehicles represented in the state and action spaces
    num_controllable_vehicles # Number of vehicles that will be part of the action space
    models # The models for the simulation
    roadway # The roadway for the simulation
    egoid # The id of the ego vehicle
    initial_scene # Initial scene
    dt # Simulation timestep
    last_observation # Last observation of the vehicle state
    actions # Set of all actions for the mdp
    action_to_index # Dictionary mapping actions to dictionary
    action_probabilities # probability of taking each action
    expand_state_space = false # Whether or not to expand state space when using convert_s
    discount = 1.
end

# Construct an Adversarial Action space
function AdversarialDrivingMDP(scene0, models, roadway, egoid, dt; expand_state_space = false, γ = 1.)
    Nveh = length(scene0)
    Nctrl = Nveh - 1
    as, aind, aprob = construct_actions(Nctrl)
    o = nothing # last observation
    AdversarialDrivingMDP(Nveh, Nctrl, models, roadway, egoid, scene0, dt, o, as, aind, aprob, expand_state_space, γ)
end

# Dynamically creates the action space based on the number of adversarial vehicles
function construct_actions(num_controllable_vehicles)
    N_actions = num_controllable_vehicles*(length(ACTIONS)-1) + 1
    actions = Array{Atype}(undef, N_actions)
    action_to_index = Dict()
    action_probabilities = Array{Float64}(undef, N_actions)

    # First select the action where all of the cars do nothing
    no_disturbance = ACTIONS[1]
    do_nothing_action = [no_disturbance for i in 1:num_controllable_vehicles]
    actions[1] = do_nothing_action
    action_to_index[do_nothing_action] = 1
    action_probabilities[1] = ACTION_PROB[1]

    index = 2
    # Then loop through all vehicles and give each one the possibilities of doing an action
    for vehid in 1:num_controllable_vehicles
        for aid in 2:length(ACTIONS) # Skip the do-nothing action
            a = copy(do_nothing_action)
            a[vehid] = ACTIONS[aid]
            actions[index] = a
            action_to_index[a] = index
            action_probabilities[index] = ACTION_PROB[aid] / num_controllable_vehicles
            index += 1
        end
    end
    action_probabilities = action_probabilities ./ sum(action_probabilities)
    @assert sum(action_probabilities) == 1
    actions, action_to_index, action_probabilities
end


POMDPs.actions(mdp::AdversarialDrivingMDP) = mdp.actions
POMDPs.actions(mdp::AdversarialDrivingMDP, state::Tuple{Scene, Float64}) = actions(mdp)
POMDPs.actionindex(mdp::AdversarialDrivingMDP, a::Atype) = mdp.action_to_index[a]

action_probability(mdp::AdversarialDrivingMDP, s::Scene, a::Atype) = mdp.action_probabilities[mdp.action_to_index[a]]

# true_action_probability(mdp::AdversarialDrivingMDP, s::Scene, a::Atype) = mdp.action_probabilities[mdp.action_to_index[a]]

# random_action(mdp::AdversarialDrivingMDP, s::Scene, rng::AbstractRNG) = mdp.actions[rand(rng, Categorical(mdp.action_probabilities))]

# Converts from vector to state
function POMDPs.convert_s(::Type{Scene}, s::AbstractArray{Float64}, mdp::AdversarialDrivingMDP)
    new_scene = Scene(Entity{BlinkerState, VehicleDef, Int64})
    Nveh = Int(length(s) / OBS_PER_VEH)

    # Loop through the vehicles in the scene, apply action and add to next scene
    for i = 1:Nveh
        j = (i-1)*OBS_PER_VEH + 1
        d = s[j] # Distance along the lane
        v = s[j+1] # velocity
        g = s[j+2] # Goal (lane id)
        b = s[j+3] # blinker

        laneid = Int(g)
        lane = mdp.roadway[laneid].lanes[1]
        blinker = Bool(b)
        vs = VehicleState(Frenet(lane, d, 0.), mdp.roadway, v)
        bv = Entity(BlinkerState(vs, blinker, mdp.models[i].goals[laneid]), VehicleDef(), i)

        if !end_of_road(bv, mdp.roadway)
            push!(new_scene, bv)
        end
    end
    new_scene
end

# Converts the state of a blinker vehicle to a vector
function to_vec(veh::Entity{BlinkerState, VehicleDef, Int64})
    Float64[posf(veh).s,
            vel(veh),
            laneid(veh),
            veh.state.blinker]
end

# Converts the state of a blinker vehicle to an expanded state space representation
function to_expanded_vec(veh::Entity{BlinkerState, VehicleDef, Int64})
    one_hot = zeros(6)
    one_hot[laneid(veh)] = 1
    s = posf(veh.state).s .* one_hot
    v = vel(veh.state) .* one_hot
    v2 = v.^2
    b = veh.state.blinker .* one_hot
    Float64[one_hot..., s..., v..., v2..., b...]
end

# Convert from state to vector (either expanded or not based on AdversarialDrivingMDP.expand_state_space flag)
function POMDPs.convert_s(::Type{AbstractArray}, state::Scene, mdp::AdversarialDrivingMDP)
    obs_size, expand_fn = OBS_PER_VEH, to_vec
    if mdp.expand_state_space
        obs_size, expand_fn = (OBS_PER_VEH_EXPANDED, to_expanded_vec)
    end
    o = isnothing(mdp.last_observation) ? zeros(obs_size*mdp.num_vehicles) : deepcopy(mdp.last_observation)
    for (ind,veh) in enumerate(state)
        o[(veh.id-1)*obs_size + 1: veh.id*obs_size] .= expand_fn(veh)
    end
    mdp.last_observation = o
    o
end

# Returns the intial state of the mdp simulator
POMDPs.initialstate(mdp::AdversarialDrivingMDP, rng::AbstractRNG = Random.GLOBAL_RNG) = mdp.initial_scene

# Get the reward from the actions taken and the next state
function POMDPs.reward(mdp::AdversarialDrivingMDP, s::Scene, a::Atype, sp::Scene)
    Float64(length(sp) > 0 && ego_collides(mdp.egoid, sp))
end

# Step the scene forward by one timestep and return the next state
function step_scene(mdp::AdversarialDrivingMDP, s::Scene, actions::Atype, rng::AbstractRNG = Random.GLOBAL_RNG)
    new_scene = Scene(Entity{BlinkerState, VehicleDef, Int64})

    # Loop through the vehicles in the scene, apply action and add to next scene
    for (i, veh) in enumerate(s)
        model = mdp.models[veh.id]
        observe!(model, s, mdp.roadway, veh.id)

        # Set the action of the adversaries (all except )
        veh.id != mdp.egoid && (model.next_action = actions[veh.id])

        a = rand(rng, mdp.models[veh.id])
        vs = propagate(veh, a, mdp.roadway, mdp.dt)
        bv = Entity(vs, veh.def, veh.id)

        if !end_of_road(bv, mdp.roadway)
            push!(new_scene, bv)
        end
    end

    return new_scene
end

# The generative interface to the POMDP
function POMDPs.gen(mdp::AdversarialDrivingMDP, s::Scene, a::Atype, rng::Random.AbstractRNG = Random.GLOBAL_RNG)
    # Simulate the scene forward one timestep
    # Try to use the existing simulate function
    sp = step_scene(mdp, s, a, rng)

    # Get the reward
    r = reward(mdp, s, a, sp)

    # Extract the observations
    # o = convert_s(Array{Float64,1}, sp, mdp)

    # Return
    (sp=sp, r=r)
end

# Discount factor for the POMDP (Set to 1 because of the finite horizon)
POMDPs.discount(mdp::AdversarialDrivingMDP) = mdp.discount

# The simulation is terminal if there is collision with the ego vehicle or if the maximum simulation time has been reached
POMDPs.isterminal(mdp::AdversarialDrivingMDP, s::Scene) = length(s) == 0 || any_collides(s)
