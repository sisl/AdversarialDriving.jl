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

