using AdversarialDriving
using AutomotiveSimulator
using POMDPSimulators
using POMDPPolicies
using POMDPs
using Test

# Construct a 3-car MDP (with and without state expansion)
scene = Scene([left_straight(1), right_turnleft(2), ego(3)])
models = Dict(i => TIDM(TIDM_template) for i=1:3)
egoid = 3
dt = 0.18
mdp = AdversarialDrivingMDP(scene, models, Tint_roadway, egoid, dt)
mdp_exp = AdversarialDrivingMDP(scene, models, Tint_roadway, egoid, dt, expand_state_space = true)


# Test MDP members
@test mdp.num_vehicles == 3
@test mdp.num_controllable_vehicles == 2
@test mdp.models == models
@test mdp.egoid == egoid
@test mdp.dt == dt
@test isnothing(mdp.last_observation)
@test discount(mdp) == 1.
@test mdp.discount == 1.
@test mdp.expand_state_space == false
@test mdp_exp.expand_state_space == true
@test initialstate(mdp) == scene

# Test action construction, probabilities and indexing
@test length(mdp.actions) == 13
@test mdp.actions[1] == [ACTIONS[1], ACTIONS[1]]
@test mdp.actions[2] == [ACTIONS[2], ACTIONS[1]]
@test mdp.actions[7] == [ACTIONS[7], ACTIONS[1]]
@test mdp.actions[8] == [ACTIONS[1], ACTIONS[2]]
@test mdp.actions[13] == [ACTIONS[1], ACTIONS[7]]

for i=1:13
    @test actions(mdp)[i] == mdp.actions[i]
    @test actionindex(mdp, actions(mdp)[i]) == i
end
@test isapprox(ACTION_PROB[1], action_probability(mdp, scene, actions(mdp)[1]))
aprob2 = action_probability(mdp, scene, actions(mdp)[2])
aprob3 = action_probability(mdp, scene, actions(mdp)[3])
@test isapprox(aprob3/aprob2, 10.)


# Test convert_s
svec = convert_s(AbstractArray, scene, mdp)
@test length(svec) == 4*3
for i=1:3
    veh = scene[i]
    b = 4*(i-1)
    @test posf(veh).s == svec[b + 1]
    @test vel(veh) == svec[b + 2]
    @test laneid(veh) == svec[b + 3]
    @test veh.state.blinker == svec[b + 4]
end

svec_exp = convert_s(AbstractArray, scene, mdp_exp)
@test length(svec_exp) == 90
@test sum(svec_exp .== 0) == 76 # One nonzero out of every 6 (extra zeros for blinkers)

# Convert the scene back
scene_back = convert_s(Scene, svec, mdp)
for i=1:length(scene)
    veh, veh_back = scene[i], scene_back[i]
    @test all(isapprox.(posg(veh), posg(veh_back)))
    @test isapprox(vel(veh), vel(veh_back))
    @test laneid(veh) == laneid(veh_back)
    @test veh.state.blinker == veh.state.blinker
end

# Test reward, isterminal
empty_scene = Scene(Entity{BlinkerState, VehicleDef, Int64})
coll_scene = Scene([left_straight(1, s=50.), right_turnleft(2, s=50.), ego(3)])
ego_coll_scene = Scene([left_straight(1, s=50.), right_turnleft(2), ego(3, s=50.)])

@test !isterminal(mdp, scene)
@test isterminal(mdp, empty_scene)
@test isterminal(mdp, coll_scene)
@test isterminal(mdp, ego_coll_scene)

@test reward(mdp, scene, mdp.actions[1], coll_scene) == 0.
@test reward(mdp, scene, mdp.actions[1], ego_coll_scene) == 1.0

# Test gen using action comparisons
sp, r = gen(mdp, scene, mdp.actions[1])
sp2, r2 = gen(mdp, scene, mdp.actions[2]) # Slowdown of first adversary
@test vel(get_by_id(sp, 1)) > vel(get_by_id(sp2, 1))


# Run full simulation with random policy
mdp.dt = 0.1
scene = Scene([left_straight(1), right_turnleft(2), ego(3, s=35., vel=12.)])
mdp.initial_scene = scene
hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> mdp.actions[1]))
@test length(hist) == 81