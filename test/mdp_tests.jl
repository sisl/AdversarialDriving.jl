using AdversarialDriving
using AutomotiveSimulator
using AutomotiveVisualization
using POMDPSimulators
using POMDPPolicies
using POMDPs
using Test

## Test construction of a BV
bv1 = BlinkerVehicleAgent(up_left(id = 10), TIDM(Tint_TIDM_template))
@test bv1.initial_entity.state isa BlinkerState
@test bv1.model.idm == Tint_TIDM_template.idm
@test bv1.num_obs == BLINKERVEHICLE_OBS
@test length(bv1.to_vec(bv1.initial_entity)) == BLINKERVEHICLE_OBS
@test bv1.to_vec(bv1.initial_entity)  == bv1.to_vec(bv1.to_entity(bv1.to_vec(bv1.initial_entity), id(bv1), Tint_roadway, bv1.model))
@test bv1.actions == BV_ACTIONS
@test bv1.action_prob == BV_ACTION_PROB
@test id(bv1) == 10


## Test construction of a Adversarial Pedestrian
ped1 = NoisyPedestrianAgent(ez_pedestrian(id=2, s=0., v=4.), AdversarialPedestrian())
@test ped1.initial_entity.state isa NoisyPedState
@test ped1.num_obs == PEDESTRIAN_OBS
@test length(ped1.to_vec(ped1.initial_entity)) == PEDESTRIAN_OBS
@test ped1.to_vec(ped1.initial_entity)  == ped1.to_vec(ped1.to_entity(ped1.to_vec(ped1.initial_entity), id(ped1), ped_roadway, ped1.model))
@test ped1.actions == []
@test ped1.action_prob == []
@test id(ped1) == 2

# #Test construction of a full MDP with BV in the Tintersection
bv2 = BlinkerVehicleAgent(left_straight(id=2), TIDM(Tint_TIDM_template))
bv3 = BlinkerVehicleAgent(right_turnleft(id=3), TIDM(Tint_TIDM_template))
bv4 = BlinkerVehicleAgent(left_turnright(id=4, s=40.), TIDM(Tint_TIDM_template))
mdp = AdversarialDrivingMDP(bv1, [bv2, bv3, bv4], Tint_roadway, 0.1)

@test agents(mdp) == [adversaries(mdp)..., sut(mdp)]
@test sutid(mdp) == 10
@test length(agents(mdp)) == 4
@test model(mdp, 10) == agents(mdp)[4].model
@test model(mdp, 2) == agents(mdp)[1].model
@test model(mdp, 3) == agents(mdp)[2].model
@test model(mdp, 4) == agents(mdp)[3].model
@test mdp.num_adversaries == 3
@test mdp.roadway == Tint_roadway
@test mdp.initial_scene[1] == bv2.initial_entity
@test mdp.initial_scene[2] == bv3.initial_entity
@test mdp.initial_scene[3] == bv4.initial_entity
@test mdp.initial_scene[4] == bv1.initial_entity
@test mdp.dt == 0.1
@test mdp.last_observation == Float64[]
@test length(convert_s(AbstractArray, mdp.initial_scene, mdp)) == 16
@test mdp.last_observation == convert_s(AbstractArray, mdp.initial_scene, mdp)
acts, action_id, action_prob = construct_discrete_actions(collect(adversaries(mdp)))
@test mdp.actions == acts
@test actions(mdp) == mdp.actions
@test mdp.action_to_index == action_id
@test actionindex(mdp, acts[4]) == 4
@test mdp.action_probabilities == action_prob
action_probability(mdp, initialstate(mdp), acts[6])
@test action_probability(mdp, initialstate(mdp), acts[6]) == action_prob[6]
@test mdp.γ == discount(mdp)
@test mdp.γ == 1.

## Test the update_adversary! function
bv5 = BlinkerVehicleAgent(left_straight(id=5), TIDM(Tint_TIDM_template))
s = initialstate(mdp)
sbefore = deepcopy(s)

update_adversary!(bv5, actions(mdp)[2][1], s)
@test all([noise(s[i]) == noise(sbefore[i]) for i=1:4])

noise_action = BlinkerVehicleControl(noise = Noise((0.2, 0.3), 1.))
update_adversary!(agents(mdp)[1], noise_action, s).state.noise.pos
@test noise(s[1]).pos == noise_action.noise.pos
@test noise(s[1]).vel == noise_action.noise.vel
@test model(mdp, id(agents(mdp)[1])).next_action == noise_action

## Test reward, isterminal
empty_scene = Scene(Entity{BlinkerState, VehicleDef, Int64})
coll_scene = Scene([left_straight(id=1, s=50.), right_turnleft(id=2, s=50.), up_left(id=10)])
ego_coll_scene = Scene([left_straight(id=1, s=50.), right_turnleft(id=2), up_left(id=10, s=50.)])

@test !isterminal(mdp, s)
@test isterminal(mdp, empty_scene)
@test isterminal(mdp, coll_scene)
@test isterminal(mdp, ego_coll_scene)

@test reward(mdp, s, actions(mdp)[1], coll_scene) == 0.
@test reward(mdp, s, actions(mdp)[1], ego_coll_scene) == 1.0

# Test gen using action comparisons
sp, r = gen(mdp, s, actions(mdp)[1])
sp2, r2 = gen(mdp, s, actions(mdp)[2]) # Slowdown of first adversary
@test vel(get_by_id(sp, 2)) > vel(get_by_id(sp2, 2))


# Run full simulation with random policy
mdp.dt = 0.1
hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> actions(mdp)[1]))
@test length(hist) == 15
# nticks = length(hist)
# timestep = mdp.dt
# scenes = state_hist(hist)
# using Reel
# animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
#     i = Int(floor(t/dt)) + 1
#     render([Tint_roadway, crosswalk, scenes[i]], canvas_width=1200, canvas_height=800)
# end
# write("ped_roadway_animated.gif", animation)

