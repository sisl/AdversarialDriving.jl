@with_kw mutable struct Agent
    initial_entity::Entity # The initial entity
    model::DriverModel # The driver model associated with this agent
    entity_dim::Int # The dimension of the entity
    disturbance_dim::Int # The disturbance dimension of this agent
    entity_to_vec::Function # A Function that converts the agent to a vector of length o_dim
    disturbance_to_vec::Union{Function, Nothing} = nothing # A Function that converts an agent action to a vector of length a_dim
    vec_to_entity::Union{Function, Nothing} = nothing # A Function that converts a vector of length o_dim to an entity
    vec_to_disturbance::Union{Function, Nothing} = nothing # A Function that converts a vector of length a_dim to an action
    actions::Array{Disturbance} = [] # List of possible actions for this agent (adversaries only)
    action_prob::Array{Float64} = [] # the associated action probabilities
end

id(a::Agent) = a.initial_entity.id

# Construct a regular Blinker vehicle agent
#TODO
function BlinkerVehicleAgent(veh::Entity{BlinkerState, D, I}, model::TIDM;
    entity_dim = BLINKERVEHICLE_ENTITY_DIM,
    disturbance_dim=BLINKERVEHICLE_DISTURBANCE_DIM,
    entity_to_vec = BlinkerVehicle_to_vec,
    disturbance_to_vec = BlinkerVehicleControl_to_vec,
    vec_to_entity = vec_to_BlinkerVehicle,
    vec_to_disturbance = vec_to_BlinkerVehicleControl,
    actions = BV_ACTIONS,
    action_prob = BV_ACTION_PROB) where {D,I}
    Agent(veh, model, entity_dim, disturbance_dim, entity_to_vec,
          disturbance_to_vec,  vec_to_entity, vec_to_disturbance, actions,
          action_prob)
end

# Construct a regular adversarial pedestrian agent
# TODO
function NoisyPedestrianAgent(ped::Entity{NoisyPedState, D, I}, model::AdversarialPedestrian;
    entity_dim = PEDESTRIAN_ENTITY_DIM,
    disturbance_dim = PEDESTRIAN_DISTURBANCE_DIM,
    entity_to_vec = NoisyPedestrian_to_vec,
    disturbance_to_vec = PedestrianControl_to_vec,
    vec_to_entity = vec_to_NoisyPedestrian_fn(DEFAULT_CROSSWALK_LANE),
    vec_to_disturbance = vec_to_PedestrianControl) where {D, I}
    Agent(ped, model, entity_dim, disturbance_dim, entity_to_vec,
          disturbance_to_vec,  vec_to_entity, vec_to_disturbance, [],[])
end

# Definition of the adversarial driving mdp
mutable struct AdversarialDrivingMDP <: MDP{Scene, Array{Disturbance}}
    agents::Array{Agent} # All the agents ordered by veh_id
    vehid2ind::Dict{Int64, Int64} # Dictionary that maps vehid to index in agent list
    num_adversaries::Int64 # The number of adversaries
    roadway::Roadway # The roadway for the simulation
    initial_scene::Scene # Initial scene
    dt::Float64 # Simulation timestep
    last_observation::Array{Float64} # Last observation of the vehicle state
    actions::Array{Array{Disturbance}} # Set of all actions for the mdp
    action_to_index::Dict{Array{Disturbance}, Int64} # Dictionary mapping actions to index
    action_probabilities::Array{Float64} # probability of taking each action
    γ::Float64 # discount
    ast_reward::Bool # A function that gives action log prob.
    no_collision_penalty::Float64 # penalty for not getting a collision (for ast reward)
    scale_reward::Bool #whether or not to scale the AST reward
    end_of_road::Float64 # specify an early end of the road
end

# Constructor
function AdversarialDrivingMDP(sut::Agent, adversaries::Array{Agent}, road::Roadway, dt::Float64;
                               discrete = true,
                               other_agents::Array{Agent} = Agent[],
                               γ = 1,
                               ast_reward = false,
                               no_collision_penalty = 1e3,
                               scale_reward = true,
                               end_of_road = Inf,)
    agents = [adversaries..., sut, other_agents...]
    d = Dict(id(agents[i]) => i for i=1:length(agents))
    Na = length(adversaries)
    scene = Scene([a.initial_entity for a in agents])
    o = Float64[] # Last observation

    as, a2i, aprob = discrete ? construct_discrete_actions(adversaries) : (Array{Disturbance}[], Dict{Array{Disturbance}, Int64}(), Float64[])
    AdversarialDrivingMDP(agents, d, Na, road, scene, dt, o, as, a2i, aprob, γ,
                         ast_reward, no_collision_penalty, scale_reward, end_of_road)
end

# Returns the intial state of the mdp simulator
POMDPs.initialstate(mdp::AdversarialDrivingMDP, rng::AbstractRNG = Random.GLOBAL_RNG) = mdp.initial_scene

# The generative interface to the POMDP
function POMDPs.gen(mdp::AdversarialDrivingMDP, s::Scene, a::Array{Disturbance}, rng::Random.AbstractRNG = Random.GLOBAL_RNG)
    mdp.last_observation = convert_s(AbstractArray, s, mdp)
    sp = step_scene(mdp, s, a, rng)
    r = reward(mdp, s, a, sp)
    (sp=sp, r=r)
end

# Get the reward from the actions taken and the next state
function POMDPs.reward(mdp::AdversarialDrivingMDP, s::Scene, a::Array{Disturbance}, sp::Scene)
    iscollision = length(sp) > 0 && ego_collides(sutid(mdp), sp)
    if mdp.ast_reward
        isterm = isterminal(mdp, sp)
        r = (isterm && !iscollision)*(-abs(mdp.no_collision_penalty)) + log(action_probability(mdp, s, a))
        mdp.scale_reward && (r = r / mdp.no_collision_penalty)
        return r
    else
        return Float64(iscollision)
    end
end

# Discount factor for the POMDP (Set to 1 because of the finite horizon)
POMDPs.discount(mdp::AdversarialDrivingMDP) = mdp.γ

# The simulation is terminal if there is collision with the ego vehicle or if the maximum simulation time has been reached
POMDPs.isterminal(mdp::AdversarialDrivingMDP, s::Scene) = !(sutid(mdp) in s)|| any_collides(s)

# Define the set of actions, action index and probability
POMDPs.actions(mdp::AdversarialDrivingMDP) = mdp.actions
POMDPs.actions(mdp::AdversarialDrivingMDP, state::Tuple{Scene, Float64}) = actions(mdp)
POMDPs.actionindex(mdp::AdversarialDrivingMDP, a::Array{Disturbance}) = mdp.action_to_index[a]
action_probability(mdp::AdversarialDrivingMDP, a::Array{Disturbance}) = mdp.action_probabilities[mdp.action_to_index[a]]
action_probability(mdp::AdversarialDrivingMDP, s::Scene, a::Array{Disturbance}) = action_probability(mdp, a)


## Helper functions

# Step the scene forward by one timestep and return the next state
function step_scene(mdp::AdversarialDrivingMDP, s::Scene, actions::Array{Disturbance}, rng::AbstractRNG = Random.GLOBAL_RNG)
    entities = []

    # Loop through the adversaries and apply the instantaneous aspects of their disturbance
    for (adversary, action) in zip(adversaries(mdp), actions)
        update_adversary!(adversary, action, s)
    end

    # Loop through the vehicles in the scene, apply action and add to next scene
    for (i, veh) in enumerate(s)
        m = model(mdp, veh.id)
        observe!(m, s, mdp.roadway, veh.id)
        a = rand(rng, m)
        bv = Entity(propagate(veh, a, mdp.roadway, mdp.dt), veh.def, veh.id)
        !end_of_road(bv, mdp.roadway, mdp.end_of_road) && push!(entities, bv)
    end
    isempty(entities) ? Scene(typeof(sut(mdp).initial_entity)) : Scene([entities...])
end

# Returns the list of agents in the mdp
agents(mdp::AdversarialDrivingMDP) = mdp.agents

# Returns the list of adversaries in the mdp
adversaries(mdp::AdversarialDrivingMDP) = view(mdp.agents, 1:mdp.num_adversaries)

# Returns the model associated with the vehid
model(mdp::AdversarialDrivingMDP, vehid::Int) = mdp.agents[mdp.vehid2ind[vehid]].model

# Returns the system under test
sut(mdp::AdversarialDrivingMDP) = mdp.agents[mdp.num_adversaries + 1]

# Returns the sut id
sutid(mdp::AdversarialDrivingMDP) = id(sut(mdp))

function update_adversary!(adversary::Agent, action::Disturbance, s::Scene)
    index = findfirst(id(adversary), s)
    isnothing(index) && return nothing # If the adversary is not in the scene then don't update
    adversary.model.next_action = action # Set the adversaries next action
    veh = s[index] # Get the actual entity
    state_type = typeof(veh.state) # Find out the type of its state
    s[index] =  Entity(state_type(veh.state, noise = action.noise), veh.def, veh.id) # replace the entity in the scene
end

