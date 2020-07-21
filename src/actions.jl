## Convert_a functions for the mdp
# Converts from vector to an action
function POMDPs.convert_a(::Type{Vector{Disturbance}}, avec::AbstractArray{Float64}, mdp::AdversarialDrivingMDP)
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
    vec = Vector{Float64}(undef, sum([agent.disturbance_dim for agent in adversaries(mdp)]))
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
    g = Float64(bv.toggle_goal)
    b = Float64(bv.toggle_blinker)
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

## Continuous action spaces
get_actions(m::Vector{Sampleable}) = error("Error! `actions` called on continuous action space")
get_actionindex(m::Vector{Sampleable}, a::Vector{Disturbance}) = error("Error! `actionindex` called on continuous action space")
function Distributions.logpdf(m::Vector{Sampleable}, a::Vector{Disturbance}, mdp::AdversarialDrivingMDP)
    avec = convert_a(AbstractArray, a, mdp)
    sum([logpdf(m[i], avec[i]) for i=1:length(avec)])
end
Base.rand(rng::AbstractRNG, m::Vector{Sampleable}, mdp::AdversarialDrivingMDP) = convert_a(Vector{Disturbance}, [rand(rng, d) for d in m], mdp)
Base.rand(rng::AbstractRNG, m::Vector{Sampleable}, convert_fn::Function) = convert_fn([rand(rng, d) for d in m])


## Discrete BlinkerVehicle Actions
struct DiscreteActionModel
    actions::Vector{Vector{Disturbance}}
    actionindex::Dict{Vector{Disturbance}, Int}
    probs::Vector{Float64}
end

function DiscreteActionModel(actions::Vector{Vector{Disturbance}}, probs::Vector{Float64})
    aind = Dict{Vector{Disturbance}, Int}(a => i for (a, i) in zip(actions, 1:length(actions)))
    DiscreteActionModel(actions, aind, probs)
end



get_actions(m::DiscreteActionModel) = m.actions
get_actionindex(m::DiscreteActionModel, a::Vector{Disturbance}) = m.actionindex[a]
Distributions.logpdf(m::DiscreteActionModel, a::Vector{Disturbance}, mdp::AdversarialDrivingMDP) = log(m.probs[findfirst(m.actions .== [a])])
Base.rand(rng::AbstractRNG, m::DiscreteActionModel, mdp::AdversarialDrivingMDP) = m.actions[rand(rng, Categorical(m.probs))]

## Combine disturbance_models
# Automatically determine the type of disturbance (continuous or discrete)
combine_disturbance_models(adversaries::Vector{Agent}) = adversaries[1].disturbance_model isa DiscreteActionModel ? combine_discrete(adversaries) : combine_continuous(adversaries)

# Combines continuous disturbance models
function combine_continuous(adversaries::Vector{Agent})
    m = Vector{Sampleable}()
    for a in adversaries
        push!(m, a.disturbance_model...)
    end
    m
end

# Combines discrete disturbance models
function combine_discrete(adversaries::Vector{Agent})
    actions = Vector{Disturbance}[]
    action_probabilities = Float64[]

    # Add the baseline action where all agents do nothing
    base_action = Disturbance[a.disturbance_model.actions[1][1] for a in adversaries]
    aprob = sum([a.disturbance_model.probs[1] for a in adversaries])
    push!(actions, base_action)
    push!(action_probabilities, aprob)

    # Loops through the rest of the actions
    for adv_i=1:length(adversaries)
        adv = adversaries[adv_i]
        for (act, prob) in zip(adv.disturbance_model.actions[2:end], adv.disturbance_model.probs[2:end])
            new_action = deepcopy(base_action)
            new_action[adv_i] = act[1]
            push!(actions, new_action)
            push!(action_probabilities, prob)
        end
    end
    action_probabilities ./= sum(action_probabilities)
    DiscreteActionModel(actions, action_probabilities)
end

## default actions
# Function to setup default actions for the blinker verhicle
get_bv_actions(med_accel = 1.5, large_accel = 3.0, med_prob = 1e-2, large_prob = 1e-3) = DiscreteActionModel(
                      Vector{Vector{Disturbance}}([ [BlinkerVehicleControl(0, 0., false, false, Noise())],
                        [BlinkerVehicleControl(0, -large_accel, false, false, Noise())],
                        [BlinkerVehicleControl(0, -med_accel, false, false, Noise())],
                        [BlinkerVehicleControl(0, med_accel, false, false, Noise())],
                        [BlinkerVehicleControl(0, large_accel, false, false, Noise())],
                        [BlinkerVehicleControl(0, 0., true, false, Noise())], # toggle goal
                        [BlinkerVehicleControl(0, 0., false, true, Noise())] # toggle blinker
                       ]),
                        [1 - (4*large_prob + 2*med_prob), large_prob, med_prob, med_prob, large_prob, large_prob, large_prob]
                        )

function get_rand_bv_actions(rng; med_accel = 1.5, large_accel = 3.0)
    probs = 10 .^ [rand(rng, Uniform(-4, -2)),
                   rand(rng, Uniform(-3, -1)),
                   rand(rng, Uniform(-3, -1)),
                   rand(rng, Uniform(-4, -2)),
                   rand(rng, Uniform(-4, -2)),
                   rand(rng, Uniform(-4, -2))]
    leftover = 1.0 - sum(probs)
    DiscreteActionModel( Vector{Vector{Disturbance}}([ [BlinkerVehicleControl(0, 0., false, false, Noise())],
                            [BlinkerVehicleControl(0, -large_accel, false, false, Noise())],
                            [BlinkerVehicleControl(0, -med_accel, false, false, Noise())],
                            [BlinkerVehicleControl(0, med_accel, false, false, Noise())],
                            [BlinkerVehicleControl(0, large_accel, false, false, Noise())],
                            [BlinkerVehicleControl(0, 0., true, false, Noise())], # toggle goal
                            [BlinkerVehicleControl(0, 0., false, true, Noise())] # toggle blinker
                           ]),
                            [leftover, probs...]
                            )
end



# TODO Add default distributions as needed

get_ped_actions(accel = 1., noise_pos = 1., p = 1e-2) = DiscreteActionModel(
                      Vector{Vector{Disturbance}}([
                        [PedestrianControl()],
                        [PedestrianControl(da = VecE2(accel, 0.))],
                        [PedestrianControl(da = VecE2(-accel, 0.))],
                        [PedestrianControl(da = VecE2(0., accel))],
                        [PedestrianControl(da = VecE2(0., -accel))],
                        # [PedestrianControl(noise = Noise(pos = VecE2(-noise_pos, 0.)))],
                        # [PedestrianControl(noise = Noise(pos = VecE2(noise_pos, 0.)))],
                        # [PedestrianControl(noise = Noise(pos = VecE2(0., noise_pos)))],
                        # [PedestrianControl(noise = Noise(pos = VecE2(0., -noise_pos)))],
                       ]),
                        [1 - (4*p), p, p, p, p]#, p, p, p, p]
                        )

