## Convert_s functions for the mdp
# Converts from vector to a Scene
function POMDPs.convert_s(::Type{Scene}, s::AbstractArray{Float64}, mdp::AdversarialDrivingMDP)
    entities = []

    # Loop through all the agents of the mdp
    index = 1
    for agent in agents(mdp)
        ent = agent.vec_to_entity(s[index : index + agent.entity_dim - 1], id(agent), mdp.roadway, agent.model)
        index += agent.entity_dim
        !end_of_road(ent, mdp.roadway) && push!(entities, ent)
    end
    isempty(entities) ? Scene(typeof(sut(mdp).initial_entity)) : Scene([entities...])
end

# Convert from Scene to a vector
function POMDPs.convert_s(::Type{AbstractArray}, state::Scene, mdp::AdversarialDrivingMDP)
    isempty(mdp.last_observation) && (mdp.last_observation = zeros(sum([a.entity_dim for a in agents(mdp)])))
    index = 1
    for a in agents(mdp)
        entity = get_by_id(state, id(a))
        mdp.last_observation[index : index + a.entity_dim - 1] .= a.entity_to_vec(entity)
        index += a.entity_dim
    end
    copy(mdp.last_observation)
end

POMDPs.convert_s(::Type{Array{Float64, 1}}, state::Scene, mdp::AdversarialDrivingMDP) = convert_s(AbstractArray, state, mdp)


## Adversarial Pedestrians vehicles
const PEDESTRIAN_ENTITY_DIM = 4

# Converts state of a pedestrian to a vector
function NoisyPedestrian_to_vec(ped::Entity{NoisyPedState, VehicleDef, Int64})
    Float64[posg(ped)..., vel(ped)]
end

function vec_to_NoisyPedestrian_fn(crosswalk_id::Int)
    function vec_to_NoisyPedestrian(arr::AbstractArray, id, roadway::Roadway, model)
        @assert length(arr) == PEDESTRIAN_ENTITY_DIM

        pos = VecSE2(arr[1], arr[2], arr[3]) # Distance along the lane
        v = arr[4]

        vs = VehicleState(pos, roadway_lane(roadway, crosswalk_id), roadway, v)
        bv = Entity(NoisyPedState(vs, Noise()), PEDESTRIAN_DEF, id)
    end
end


## Blinker vehicles
const BLINKERVEHICLE_ENTITY_DIM = 4
const BLINKERVEHICLE_EXPANDED_ENTITY_DIM = 30

# Converts the state of a blinker vehicle to a vector
function BlinkerVehicle_to_vec(veh::Entity{BlinkerState, VehicleDef, Int64})
    Float64[posf(veh).s, vel(veh), laneid(veh), veh.state.blinker]
end

# Converts the BlinkerVehicle vector back to an agent
function vec_to_BlinkerVehicle(arr::AbstractArray, id, roadway::Roadway, model)
    @assert length(arr) == BLINKERVEHICLE_ENTITY_DIM
    s = arr[1] # Distance along the lane
    v = arr[2] # velocity
    g = Int(arr[3]) # Goal (lane id)
    b = Bool(arr[4]) # blinker

    vs = VehicleState(Frenet(roadway, g, s), roadway, v)
    bs = BlinkerState(vs, b, model.goals[g], Noise())
    bv = Entity(bs, VehicleDef(), id)
end

# Converts the state of a blinker vehicle to an expanded state space representation
function BlinkerVehicle_to_expanded_vec(veh::Entity{BlinkerState, VehicleDef, Int64})
    one_hot = zeros(6)
    one_hot[laneid(veh)] = 1
    s = posf(veh.state).s .* one_hot
    v = vel(veh.state) .* one_hot
    v2 = v.^2
    b = veh.state.blinker .* one_hot
    Float64[one_hot..., s..., v..., v2..., b...]
end

