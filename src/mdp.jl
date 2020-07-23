@with_kw mutable struct Agent
    get_initial_entity::Function # Returns an entity IC
    model::DriverModel # The driver model associated with this agent
    entity_dim::Int # The dimension of the entity
    disturbance_dim::Int # The disturbance dimension of this agent
    entity_to_vec::Function # A Function that converts the agent to a vector of length o_dim
    disturbance_to_vec::Union{Function, Nothing} = nothing # A Function that converts an agent action to a vector of length a_dim
    vec_to_entity::Union{Function, Nothing} = nothing # A Function that converts a vector of length o_dim to an entity
    vec_to_disturbance::Union{Function, Nothing} = nothing # A Function that converts a vector of length a_dim to an action
    disturbance_model = nothing  # Model of the disturbances (supports logpdf, rand and actions)
end

id(a::Agent) = a.get_initial_entity().id

# Construct a regular Blinker vehicle agent
function BlinkerVehicleAgent(get_veh::Function, model::TIDM;
    entity_dim = BLINKERVEHICLE_ENTITY_DIM,
    disturbance_dim=BLINKERVEHICLE_DISTURBANCE_DIM,
    entity_to_vec = BlinkerVehicle_to_vec,
    disturbance_to_vec = BlinkerVehicleControl_to_vec,
    vec_to_entity = vec_to_BlinkerVehicle,
    vec_to_disturbance = vec_to_BlinkerVehicleControl,
    disturbance_model = get_bv_actions())
    Agent(get_veh, model, entity_dim, disturbance_dim, entity_to_vec,
          disturbance_to_vec,  vec_to_entity, vec_to_disturbance, disturbance_model)
end

# Construct a regular adversarial pedestrian agent
function NoisyPedestrianAgent(get_ped::Function, model::AdversarialPedestrian;
    entity_dim = PEDESTRIAN_ENTITY_DIM,
    disturbance_dim = PEDESTRIAN_DISTURBANCE_DIM,
    entity_to_vec = NoisyPedestrian_to_vec,
    disturbance_to_vec = PedestrianControl_to_vec,
    vec_to_entity = vec_to_NoisyPedestrian_fn(DEFAULT_CROSSWALK_LANE),
    vec_to_disturbance = vec_to_PedestrianControl,
    disturbance_model = get_ped_actions())
    Agent(get_ped, model, entity_dim, disturbance_dim, entity_to_vec,
          disturbance_to_vec,  vec_to_entity, vec_to_disturbance, disturbance_model)
end

# Definition of the adversarial driving mdp
mutable struct AdversarialDrivingMDP <: MDP{Scene, Vector{Disturbance}}
    agents::Vector{Agent} # All the agents ordered by (adversaries..., sut, others...)
    vehid2ind::Dict{Int64, Int64} # Dictionary that maps vehid to index in agent list
    num_adversaries::Int64 # The number of adversaries
    roadway::Roadway # The roadway for the simulation
    dt::Float64 # Simulation timestep
    last_observation::Array{Float64} # Last observation of the vehicle state
    disturbance_model # Model used for disturbances. supports `logpdf` and `rand` and `actions`
    γ::Float64 # discount
    ast_reward::Bool # A function that gives action log prob.
    no_collision_penalty::Float64 # penalty for not getting a collision (for ast reward)
    scale_reward::Bool #whether or not to scale the AST reward
    end_of_road::Float64 # specify an early end of the road
end

# Constructor
function AdversarialDrivingMDP(sut::Agent, adversaries::Vector{Agent}, road::Roadway, dt::Float64;
                               other_agents::Vector{Agent} = Agent[],
                               γ = 1,
                               ast_reward = false,
                               no_collision_penalty = 1e3,
                               scale_reward = true,
                               end_of_road = Inf,)
    agents = [adversaries..., sut, other_agents...]
    d = Dict(id(agents[i]) => i for i=1:length(agents))
    Na = length(adversaries)
    o = Float64[] # Last observation

    m = combine_disturbance_models(adversaries)
    AdversarialDrivingMDP(agents, d, Na, road, dt, o, m, γ,
                         ast_reward, no_collision_penalty, scale_reward, end_of_road)
end

# Returns the intial state of the mdp simulator
function POMDPs.initialstate(mdp::MDP{Scene, A}, rng::AbstractRNG = Random.GLOBAL_RNG) where A
     Scene([a.get_initial_entity(rng) for a in agents(mdp)])
 end

# The generative interface to the POMDP
function POMDPs.gen(mdp::MDP{Scene, A}, s::Scene, a::A, rng::Random.AbstractRNG = Random.GLOBAL_RNG) where A
    mdp.last_observation = convert_s(AbstractArray, s, mdp)
    sp = step_scene(mdp, s, a, rng)
    r = reward(mdp, s, a, sp)
    (sp=sp, r=r)
end

# Get the reward from the actions taken and the next state
function POMDPs.reward(mdp::AdversarialDrivingMDP, s::Scene, a::Vector{Disturbance}, sp::Scene)
    iscollision = length(sp) > 0 && ego_collides(sutid(mdp), sp)
    if mdp.ast_reward
        isterm = isterminal(mdp, sp)
        r = logpdf(mdp, s, a)
        r += iscollision * abs(mdp.no_collision_penalty)
        # r = (isterm && !iscollision)*(-abs(mdp.no_collision_penalty)) + logpdf(mdp, s, a)
        mdp.scale_reward && (r = r / abs(mdp.no_collision_penalty))
        return Float32(r)
    else
        return Float32(iscollision)
    end
end

# Discount factor for the POMDP (Set to 1 because of the finite horizon)
POMDPs.discount(mdp::MDP{Scene, A}) where A = mdp.γ

# The simulation is terminal if there is collision with the ego vehicle or if the maximum simulation time has been reached
POMDPs.isterminal(mdp::MDP{Scene, A}, s::Scene) where A = !(sutid(mdp) in s)|| any_collides(s)

# Define the set of actions, action index and probability
POMDPs.actions(mdp::AdversarialDrivingMDP) = get_actions(mdp.disturbance_model)
POMDPs.actionindex(mdp::AdversarialDrivingMDP, a::Vector{Disturbance}) = Int32(get_actionindex(mdp.disturbance_model, a))

# The default disturbance policy according to the disturbance models
#TODO: Switch the "Function policy" to something that can get logpdf?
default_policy(mdp::AdversarialDrivingMDP, rng::AbstractRNG = Random.GLOBAL_RNG) = FunctionPolicy((s) -> rand(rng, mdp.disturbance_model))

# POMDPs.actionindex(mdp::AdversarialDrivingMDP, a::Vector{Disturbance}) = findfirst(actions(mdp) .== a)
Distributions.logpdf(mdp::AdversarialDrivingMDP, a::Vector{Disturbance}) = logpdf(mdp.disturbance_model, a, mdp)
Distributions.logpdf(mdp::AdversarialDrivingMDP, s::Scene, a::Vector{Disturbance}) = logpdf(mdp, a)
Distributions.logpdf(mdp::AdversarialDrivingMDP, h::SimHistory) = sum([logpdf(mdp, s, a) for (s,a) in eachstep(h, (:s, :a))])
Distributions.logpdf(mdp::AdversarialDrivingMDP, as::Vector) = sum([logpdf(mdp, a) for a in as])

## Helper functions

# Step the scene forward by one timestep and return the next state
function step_scene(mdp::AdversarialDrivingMDP, s::Scene, actions::Vector{Disturbance}, rng::AbstractRNG = Random.GLOBAL_RNG)
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
    isempty(entities) ? Scene(typeof(sut(mdp).get_initial_entity())) : Scene([entities...])
end

# Returns the list of agents in the mdp
agents(mdp::MDP{Scene,A}) where A = mdp.agents

# Returns the list of adversaries in the mdp
adversaries(mdp::MDP{Scene,A}) where A = view(mdp.agents, 1:mdp.num_adversaries)

# Returns the model associated with the vehid
model(mdp::MDP{Scene,A}, vehid::Int) where A = mdp.agents[mdp.vehid2ind[vehid]].model

# Returns the system under test
sut(mdp::MDP{Scene,A}) where A  = mdp.agents[mdp.num_adversaries + 1]

# Returns the sut id
sutid(mdp::MDP{Scene,A}) where A = id(sut(mdp))

function update_adversary!(adversary::Agent, action::Disturbance, s::Scene)
    index = findfirst(id(adversary), s)
    isnothing(index) && return nothing # If the adversary is not in the scene then don't update
    adversary.model.next_action = action # Set the adversaries next action
    veh = s[index] # Get the actual entity
    state_type = typeof(veh.state) # Find out the type of its state
    s[index] =  Entity(state_type(veh.state, noise = action.noise), veh.def, veh.id) # replace the entity in the scene
end


## SUT driving MDP
mutable struct DrivingMDP <: MDP{Scene, BlinkerVehicleControl}
    agents::Vector{Agent} # All the agents ordered by (adversaries..., sut, others...)
    vehid2ind::Dict{Int64, Int64} # Dictionary that maps vehid to index in agent list
    num_adversaries::Int64 # The number of adversaries
    roadway::Roadway # The roadway for the simulation
    dt::Float64 # Simulation timestep
    last_observation::Array{Float64} # Last observation of the vehicle state
    γ::Float64 # discount
    end_of_road::Float64 # Early stopping of road
    per_timestep_penalty::Float64
    v_des::Float64
end

# Constructor
function DrivingMDP(sut::Agent, adversaries::Vector{Agent}, road::Roadway, dt::Float64; γ = 1, end_of_road = Inf, per_timestep_penalty = 0, v_des = 25)
    agents = [adversaries..., sut]
    d = Dict(id(agents[i]) => i for i=1:length(agents))
    DrivingMDP(agents, d, length(adversaries), road, dt, Float64[], γ, end_of_road, per_timestep_penalty, v_des)
end

# Get the reward from the actions taken and the next state
function POMDPs.reward(mdp::DrivingMDP, s::Scene, a::BlinkerVehicleControl, sp::Scene)
    id = sutid(mdp)
    v = vel(get_by_id(s, id))

    r = -abs(mdp.per_timestep_penalty)
    # If the simulation ends but the SUT is not at the end of the road, big penalty
    if ego_collides(id, sp)
        r += -1
    elseif isterminal(mdp, sp)
        r += 1
    end
    r += -.1 * (v < 0)
    r += -0.001 * abs(v  - mdp.v_des)
    r
end

function step_scene(mdp::DrivingMDP, s::Scene, action::BlinkerVehicleControl, rng::AbstractRNG = Random.GLOBAL_RNG)
    entities = []
    sid = sutid(mdp)

    # Choose random actions for the adversaries
    for adversary in adversaries(mdp)
        adv_action = rand(rng, adversary.disturbance_model, adversary.vec_to_disturbance)
        update_adversary!(adversary, adv_action, s)
    end

    # Loop through the vehicles in the scene, apply action and add to next scene
    for (i, veh) in enumerate(s)
        if veh.id == sid # for the sut, use the prescribed action
            a = action
        else # For the other vehicles use their  model
            m = model(mdp, veh.id)
            observe!(m, s, mdp.roadway, veh.id)
            a = rand(rng, m)
        end
        bv = Entity(propagate(veh, a, mdp.roadway, mdp.dt), veh.def, veh.id)
        !end_of_road(bv, mdp.roadway, mdp.end_of_road) && push!(entities, bv)
    end
    isempty(entities) ? Scene(typeof(sut(mdp).get_initial_entity())) : Scene([entities...])
end

# Define the set of actions, action index and probability
POMDPs.actions(mdp::DrivingMDP) = [BlinkerVehicleControl(a = -4.), BlinkerVehicleControl(a= -2.), BlinkerVehicleControl(a = 0.), BlinkerVehicleControl(a = 1.5), BlinkerVehicleControl(a = 3.)]
POMDPs.actionindex(mdp::DrivingMDP, a::BlinkerVehicleControl) = findfirst([a] .== actions(mdp))

