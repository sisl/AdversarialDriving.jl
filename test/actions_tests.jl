using AdversarialDriving
using Test


# Create two agents and construct their actions
bv1 = BlinkerVehicleAgent(left_straight(id=1), TIDM(Tint_TIDM_template))
bv2 = BlinkerVehicleAgent(right_straight(id=2), TIDM(Tint_TIDM_template))
advs = [bv1, bv2]
acts, action_id, action_prob = construct_discrete_actions(advs)
@test acts isa Array{Array{Disturbance}}
@test action_id isa Dict{Array{Disturbance}, Int64}
@test action_prob isa Array{Float64}

# Test action construction, probabilities and indexing
@test length(acts) == 13
@test acts[1] == [BV_ACTIONS[1], BV_ACTIONS[1]]
@test acts[2] == [BV_ACTIONS[2], BV_ACTIONS[1]]
@test acts[7] == [BV_ACTIONS[7], BV_ACTIONS[1]]
@test acts[8] == [BV_ACTIONS[1], BV_ACTIONS[2]]
@test acts[13] == [BV_ACTIONS[1], BV_ACTIONS[7]]

for i=1:13
    @test action_id[acts[i]] == i
end
@test isapprox(BV_ACTION_PROB[1], action_prob[1])
aprob2 = action_prob[2]
aprob3 = action_prob[3]
@test isapprox(aprob3/aprob2, 10.)

