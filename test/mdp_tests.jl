using AdversarialDriving
using AutomotiveSimulator
using AutomotiveVisualization
using POMDPSimulators
using POMDPPolicies
using POMDPs
using Test
using Distributions

## Test construction of a BV
bv1 = BlinkerVehicleAgent(up_left(id = 10), TIDM(Tint_TIDM_template))
@test bv1.get_initial_entity().state isa BlinkerState
@test bv1.model.idm == Tint_TIDM_template.idm
@test bv1.entity_dim == BLINKERVEHICLE_ENTITY_DIM
@test bv1.disturbance_dim == BLINKERVEHICLE_DISTURBANCE_DIM
@test length(bv1.entity_to_vec(bv1.get_initial_entity())) == BLINKERVEHICLE_ENTITY_DIM
@test bv1.entity_to_vec(bv1.get_initial_entity())  == bv1.entity_to_vec(bv1.vec_to_entity(bv1.entity_to_vec(bv1.get_initial_entity()), id(bv1), Tint_roadway, bv1.model))
@test bv1.actions == BV_ACTIONS
@test bv1.action_prob == BV_ACTION_PROB
@test id(bv1) == 10


## Test construction of a Adversarial Pedestrian
ped1 = NoisyPedestrianAgent(get_pedestrian(id=2, s=0., v=4.), AdversarialPedestrian())
@test ped1.get_initial_entity().state isa NoisyPedState
@test ped1.entity_dim == PEDESTRIAN_ENTITY_DIM
@test ped1.disturbance_dim == PEDESTRIAN_DISTURBANCE_DIM
@test length(ped1.entity_to_vec(ped1.get_initial_entity())) == PEDESTRIAN_ENTITY_DIM
@test ped1.entity_to_vec(ped1.get_initial_entity())  == ped1.entity_to_vec(ped1.vec_to_entity(ped1.entity_to_vec(ped1.get_initial_entity()), id(ped1), ped_roadway, ped1.model))
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
scene = initialstate(mdp)
@test scene[1] == bv2.get_initial_entity()
@test scene[2] == bv3.get_initial_entity()
@test scene[3] == bv4.get_initial_entity()
@test scene[4] == bv1.get_initial_entity()
@test mdp.dt == 0.1
@test mdp.last_observation == Float64[]
@test length(convert_s(AbstractArray, scene, mdp)) == 16
@test mdp.last_observation == convert_s(AbstractArray, scene, mdp)
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
@test mdp.ast_reward == false
@test mdp.no_collision_penalty == 1e3
@test mdp.scale_reward == true
@test mdp.end_of_road == Inf

mdp_temp = AdversarialDrivingMDP(bv1, [bv2, bv3, bv4], Tint_roadway, 0.1, γ=0.9, ast_reward = true, no_collision_penalty = 1e7, scale_reward = false, end_of_road = 70.)
@test mdp_temp.γ == .9
@test mdp_temp.ast_reward == true
@test mdp_temp.no_collision_penalty == 1e7
@test mdp_temp.scale_reward == false
@test mdp_temp.end_of_road == 70.


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
coll_scene = Scene([left_straight(id=1, s=50.)(), right_turnleft(id=2, s=50.)(), up_left(id=10)()])
ego_coll_scene = Scene([left_straight(id=1, s=50.)(), right_turnleft(id=2)(), up_left(id=10, s=50.)()])

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

## Test the reward function from rollouts (both with and without ast)
sut_agent = BlinkerVehicleAgent(up_left(id=1, s=25., v=15.), TIDM(Tint_TIDM_template, noisy_observations = true))
adv1 = BlinkerVehicleAgent(left_straight(id=2, s=20., v=15.0), TIDM(Tint_TIDM_template))
mdp = AdversarialDrivingMDP(sut_agent, [adv1], Tint_roadway, 0.15)
blinker_action = mdp.actions[7]
hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> posf(get_by_id(initialstate(mdp),1)).s <=25 ? blinker_action : actions(mdp)[1]))
@test undiscounted_reward(hist) == 1

mdp = AdversarialDrivingMDP(sut_agent, [adv1], Tint_roadway, 0.15, ast_reward = true)
hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> posf(get_by_id(initialstate(mdp),1)).s <=25 ? blinker_action : actions(mdp)[1]))
@test undiscounted_reward(hist) > -1

hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> actions(mdp)[1]))
@test undiscounted_reward(hist) < -1


## Test random initial states
sut_agent = BlinkerVehicleAgent(rand_up_left(id=1, s_dist=Uniform(15,35.), v_dist=Uniform(7,25.)), TIDM(Tint_TIDM_template, noisy_observations = true))
adv1 = BlinkerVehicleAgent(rand_left(id=2, s_dist=Uniform(15,35.), v_dist=Uniform(7,25.)), TIDM(Tint_TIDM_template))
mdp = AdversarialDrivingMDP(sut_agent, [adv1], Tint_roadway, 0.15)

s1 = initialstate(mdp)
s2 = initialstate(mdp)

@test posf(get_by_id(s1, 1)) != posf(get_by_id(s2, 1))
for i=1:100
    s1 = initialstate(mdp)
    @test posf(get_by_id(s1, 1)).s >= 15. && posf(get_by_id(s1, 1)).s <= 35
    @test vel(get_by_id(s1, 1)) >= 7. && vel(get_by_id(s1, 1)) <= 25.
end

## Test decompose indices
sut_agent = BlinkerVehicleAgent(up_left(id=1, s=25., v=15.), TIDM(Tint_TIDM_template, noisy_observations = true))
adv1 = BlinkerVehicleAgent(left_straight(id=2, s=0., v=15.0), TIDM(Tint_TIDM_template))
adv2 = BlinkerVehicleAgent(left_turnright(id=3, s=10., v=15.0), TIDM(Tint_TIDM_template))
adv3 = BlinkerVehicleAgent(right_straight(id=4, s=40., v=20.0), TIDM(Tint_TIDM_template))
adv4 = BlinkerVehicleAgent(right_turnleft(id=5, s=30., v=20.0), TIDM(Tint_TIDM_template))

mdp = AdversarialDrivingMDP(sut_agent, [adv1, adv2, adv3, adv4], Tint_roadway, 0.1)

@test decompose_indices(mdp, ids = [2,3]) == [1,2,3,4,5,6,7,8]
@test decompose_indices(mdp, ids = [5,1]) == [13,14,15,16,17,18,19,20]
@test decompose_indices(mdp, ids = [1,4]) == [9,10,11,12,17,18,19,20]

## Create actions from dictionary
d = Dict(:da => rand(10),
         :toggle_blinker => rand(Bernoulli(0.1), 10),
         :toggle_goal => rand(Bernoulli(0.1), 10),
         :noise_s => rand(10),
         :noise_v => rand(10))

as = create_actions_1BV(d)

for i in 1:length(as)
    a = as[i][1]
    @test a.da == d[:da][i]
    @test a.toggle_goal == d[:toggle_goal][i]
    @test a.toggle_blinker == d[:toggle_blinker][i]
    @test a.noise.pos[1] == d[:noise_s][i]
    @test a.noise.vel == d[:noise_v][i]
end


sut_agent = BlinkerVehicleAgent(up_left(id=1, s=25., v=15.), TIDM(Tint_TIDM_template, noisy_observations = true))
adv1 = BlinkerVehicleAgent(left_straight(id=2, s=20., v=15.0), TIDM(Tint_TIDM_template))
mdp = AdversarialDrivingMDP(sut_agent, [adv1], Tint_roadway, 0.15)

d = Dict(:da => Uniform(),
         :toggle_blinker => Bernoulli(0.1),
         :toggle_goal => Bernoulli(0.1),
         :noise_s => Uniform(),
         :noise_v => Uniform())

h = POMDPSimulators.simulate(HistoryRecorder(),mdp, continous_policy(d))

