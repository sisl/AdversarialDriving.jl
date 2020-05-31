## Convert_a functions for the mdp

# Converts from vector to an action
function POMDPs.convert_a(::Type{Array{Disturbance}}, avec::AbstractArray{Float64}, mdp::AdversarialDrivingMDP)
    a = Disturbance[]
    index = 1
    for agent in adversaries(mdp)
        d = agent.vec_to_disturbance(avec[index : index + agent.disturbance_dim - 1])
        index += agent.disturbance_dim
        push!(a, d)
    end
    a
end

# Convert from action to a vector
function POMDPs.convert_a(::Type{AbstractArray}, a::Vector{Disturbance}, mdp::AdversarialDrivingMDP)
    vec = Array{Float64}(undef, sum([agent.disturbance_dim for agent in adversaries(mdp)]))
    index = 1
    for i = 1:length(adversaries(mdp))
        agent = adversaries(mdp)[i]
        vec[index : index + agent.disturbance_dim - 1] .= agent.disturbance_to_vec(a[i])
        index += agent.disturbance_dim
    end
    vec
end

## PedestrianControl Disturbances
const PEDESTRIAN_DISTURBANCE_DIM = 5

# Converts PedestrianControl disturbance to a vector
function PedestrianControl_to_vec(pc::PedestrianControl)
    Float64[pc.da..., pc.noise.pos..., pc.noise.vel]
end

# Converts from a vector to a PedestrianConrol Disturbance
function vec_to_PedestrianControl(arr::AbstractArray)
    @assert length(arr) == PEDESTRIAN_DISTURBANCE_DIM
    da = VecE2(arr[1], arr[2])
    noise_pos = VecE2(arr[3], arr[4])
    noise_v = arr[5]
    PedestrianControl(VecE2(0., 0.), da, Noise(noise_pos, noise_v))
end

## BlinkerVehicle Disturbances
const BLINKERVEHICLE_DISTURBANCE_DIM = 5

# Converts PedestrianControl disturbance to a vector
function BlinkerVehicleControl_to_vec(bv::BlinkerVehicleControl)
    g = Float64(bv.toggle_goal) - 0.5
    b = Float64(bv.toggle_blinker) - 0.5
    Float64[bv.da, g, b, bv.noise.pos[1], bv.noise.vel]
end

# Converts from a vector to a PedestrianConrol Disturbance
function vec_to_BlinkerVehicleControl(arr::AbstractArray)
    @assert length(arr) == BLINKERVEHICLE_DISTURBANCE_DIM
    da = arr[1]
    toggle_goal = arr[2] > 0
    toggle_blinker = arr[3] > 0
    noise_pos = VecE2(arr[4], 0.)
    noise_v = arr[5]
    BlinkerVehicleControl(0., da, toggle_goal, toggle_blinker, Noise(noise_pos, noise_v))
end


## Discrete BlinkerVehicle Actions
const BV_ACTIONS = [ BlinkerVehicleControl(0, 0., false, false, Noise()),
                        BlinkerVehicleControl(0, -3., false, false, Noise()),
                        BlinkerVehicleControl(0, -1.5, false, false, Noise()),
                        BlinkerVehicleControl(0, 1.5, false, false, Noise()),
                        BlinkerVehicleControl(0, 3., false, false, Noise()),
                        BlinkerVehicleControl(0, 0., true, false, Noise()), # toggle goal
                        BlinkerVehicleControl(0, 0., false, true, Noise()) # toggle blinker
                        ]
const BV_ACTION_PROB = [1 - (4e-3 + 2e-2), 1e-3, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3]


# Dynamically creates the action space based on the number of adversarial vehicles
function construct_discrete_actions(adversaries::Array{Agent})
    actions = Array{Disturbance}[]
    action_to_index = Dict{Array{Disturbance}, Int64}()
    action_probabilities = Float64[]

    # Add the baseline action where all agents do nothing
    base_action = [a.actions[1] for a in adversaries]
    aprob = sum([a.action_prob[1] for a in adversaries])
    push!(actions, base_action)
    push!(action_probabilities, aprob)
    action_to_index[actions[end]] = length(actions)

    # Loops through the rest of the actions
    for adv_i=1:length(adversaries)
        adv = adversaries[adv_i]
        for (act, prob) in zip(adv.actions[2:end], adv.action_prob[2:end])
            new_action = deepcopy(base_action)
            new_action[adv_i] = act
            push!(actions, new_action)
            push!(action_probabilities, prob)
            action_to_index[actions[end]] = length(actions)
        end
    end
    action_probabilities ./= sum(action_probabilities)
    actions, action_to_index, action_probabilities
end

