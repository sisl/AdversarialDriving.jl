include("../simulator/adm_task_generator.jl")
using Test

# generate combined and decomposed POMDPs
decomposed, pomdp = generate_decomposed_scene(dt = 0.18)

# Check sizes of the constructed pomdps
@test length(decomposed) == length(pomdp.initial_scene)-1
@test length(decomposed[1].initial_scene) == 2
@test pomdp.num_controllable_vehicles == 4
@test pomdp.num_vehicles == 5
@test pomdp.dt == decomposed[1].dt
@test decomposed[2].dt == 0.18
@test o_dim(pomdp) == 5*OBS_PER_VEH
@test a_dim(3) == ACT_PER_VEH^3
@test a_dim(pomdp) == ACT_PER_VEH^pomdp.num_controllable_vehicles

# Confirm model indexing and actions
@test sort(collect(keys(pomdp.models))) == [1,2,3,4,5]
@test pomdp.egoid == 5
@test all([decomposed[i].egoid == 2 for i in 1:4])
das = support(decomposed[1].models[1].da_dist)
@test decomposed[1].actions[1] == [LaneFollowingAccelBlinker(0, das[1], false, false)]
@test decomposed[1].actions[2] == [LaneFollowingAccelBlinker(0, das[2], false, false)]
@test decomposed[1].actions[3] == [LaneFollowingAccelBlinker(0, das[3], false, false)]
@test decomposed[1].actions[4] == [LaneFollowingAccelBlinker(0, das[4], false, false)]
@test decomposed[1].actions[5] == [LaneFollowingAccelBlinker(0, das[5], false, false)]
@test decomposed[1].actions[6] == [LaneFollowingAccelBlinker(0, 0, true, false)]
@test decomposed[1].actions[7] == [LaneFollowingAccelBlinker(0, 0, false, true)]
@test length(pomdp.actions) == 7^4
for i=1:length(pomdp.actions)
    @test pomdp.action_to_index[pomdp.actions[i]] == i
    @test pomdp.action_to_index[pomdp.actions[i]] == actionindex(pomdp, actions(pomdp)[i])
end
@test actions(pomdp) == pomdp.actions

# Confirm that the action distributions are assigned correctly
@test pomdp.models[5].force_action == false
@test all([pomdp.models[i].force_action == true for i in 1:4])
@test support(decomposed[1].models[1].da_dist) == [-1.75, -0.5, 0., 0.5, 1.75]
@test support(decomposed[1].models[2].da_dist) == [-1.75, -0.5, 0., 0.5, 1.75]
@test probs(decomposed[1].models[1].da_dist) == [0.025, 0.075, 0.8, 0.075, 0.025]
@test probs(decomposed[1].models[2].da_dist) == [0,0,1,0,0]
@test decomposed[1].models[1].toggle_goal_dist == Bernoulli(1e-2)
@test decomposed[1].models[2].toggle_goal_dist == Bernoulli(0)
@test decomposed[1].models[1].da_force == 0
@test decomposed[1].models[1].toggle_blinker_force == false
@test decomposed[1].models[1].toggle_goal_force == false

# Make sure the action sampling is working correctly
observe!(decomposed[1].models[1], decomposed[1].initial_scene, decomposed[1].roadway, 2)
observe!(decomposed[1].models[2], decomposed[1].initial_scene, decomposed[1].roadway, 2)
@test rand(decomposed[1].models[1]) == LaneFollowingAccelBlinker(-9.0, 0, false, false)
@test rand(decomposed[1].models[2]) == LaneFollowingAccelBlinker(-9.0, 0, false, false)

decomposed[1].models[1].da_force = -1.75
decomposed[1].models[1].toggle_blinker_force = true
decomposed[1].models[1].toggle_goal_force = true
decomposed[1].models[2].da_force = -1.75
decomposed[1].models[2].toggle_blinker_force = true
decomposed[1].models[2].toggle_goal_force = true

@test rand(decomposed[1].models[1]) == LaneFollowingAccelBlinker(-9.0, -1.75, true, true)
@test rand(decomposed[1].models[2]) == LaneFollowingAccelBlinker(-9.0, 0, false, false)

# Test log probability
@test action_logprob(decomposed[1].models[1], LaneFollowingAccelBlinker(-9.0, -1.75, true, true)) == log(0.025) + log(1e-2) + log(1e-2)
@test action_logprob(decomposed[1].models[1], LaneFollowingAccelBlinker(-9.0, -1.8, true, true)) == log(0.025) + log(1e-2) + log(1e-2)
@test action_logprob(decomposed[1].models[1], LaneFollowingAccelBlinker(-9.0, -1, true, true)) == log(0.075) + log(1e-2) + log(1e-2)


# Test state conversion
s = initialstate(pomdp)
@test s == pomdp.initial_scene
svec = convert_s(AbstractArray, s, pomdp)
@test length(svec) == pomdp.num_vehicles*4

@test svec[1] == posf(s.entities[1].state).s
@test svec[2] == vel(s.entities[1].state)
@test svec[3] == laneid(s.entities[1])
@test svec[4] == s.entities[1].state.blinker

@test svec == convert_s(AbstractArray, convert_s(BlinkerScene, svec, pomdp), pomdp)

# try out collision checking, reward and discount
@test discount(pomdp) == 1
@test !iscollision(pomdp, s)
@test !isterminal(pomdp, s)
svec[1] = 50
svec[17] = 50
s_coll = convert_s(BlinkerScene, svec, pomdp)
@test iscollision(pomdp, s_coll)
@test isterminal(pomdp, s_coll)

svec[1] = 100
svec[5] = 100
svec[9] = 100
svec[13] = 100
svec[17] = 100
s_no_cars = convert_s(BlinkerScene, svec, pomdp)
@test length(s_no_cars) == 0
@test isterminal(pomdp, s_no_cars)


# Do a nominal rollout
policy(o) = rand(actions(pomdp))
o,a,r,s = policy_rollout(pomdp, policy, initialstate(pomdp), save_scenes = true)
@test length(s[2]) == length(s[1])

# include("../experiments/plot_utils.jl")
# make_interact(s, pomdp.models, pomdp.roadway, egoid = 5)
# make_video(s, pomdp.models, pomdp.roadway, "mc_rollout", egoid = 5)


