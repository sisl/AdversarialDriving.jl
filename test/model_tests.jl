using AdversarialDriving
using AutomotiveSimulator
using AutomotiveVisualization
using Random
using Test
using Distributions

# Check out roadways
# a = render([Tint_roadway])
# a = render([ped_roadway, crosswalk])

## Test the struct types and construction
pos_noise = Noise((0.3, 0.1), 2.)
neg_noise = Noise((-0.3, -0.1), -2.)
@test pos_noise.pos[1] == 0.3
@test pos_noise.pos[2] == 0.1
@test pos_noise.vel == 2.

ped = NoisyPedState(VehicleState(VecSE2(25, -10, π/2), ped_roadway, 3.), pos_noise)
ped_entity = Entity(ped, PEDESTRIAN_DEF, 1)
@test ped isa NoisyState
@test posg(ped) == VecSE2(25, -10, π/2)
@test vel(ped) == 3.
@test noise(ped) == pos_noise
@test laneid(ped_entity) == 2

veh = BlinkerState(VehicleState(VecSE2(50., 0., 0.), ped_roadway, 3.), true, [3,4], neg_noise)
bv_entity = Entity(veh, VehicleDef(), 1)
@test veh isa NoisyState
@test posg(veh) == VecSE2(50., 0, 0.)
@test vel(veh) == 3.
@test blinker(veh)
@test goals(veh) == [3,4]
@test noise(veh) == neg_noise
@test laneid(bv_entity) == 1

## Test getters for blinker noise and goals
@test blinker(bv_entity)
@test blinker(bv_entity.state)
@test !blinker(ped_entity)
@test !blinker(ped_entity.state)

@test goals(bv_entity) == [3,4]
@test goals(bv_entity.state) == [3,4]
@test goals(ped_entity) == [2]
@test goals(ped_entity.state) == [2]

@test noise(bv_entity) == neg_noise
@test noise(bv_entity.state) ==  neg_noise
@test noise(ped_entity) == pos_noise
@test noise(ped_entity.state) == pos_noise

## Test update_veh_state
new_state = VehicleState(VecSE2(50., 50., 0.), ped_roadway, 30.)

new_ped = update_veh_state(ped, new_state)
@test new_ped isa NoisyPedState
@test new_ped.veh_state == new_state
@test noise(new_ped) == pos_noise

new_veh = update_veh_state(veh, new_state)
@test new_veh isa BlinkerState
@test new_veh.veh_state == new_state
@test noise(new_veh) == neg_noise

## Test the noise

# Check the noisy_entity function
noisy_ped = noisy_entity(ped_entity, ped_roadway)
@test laneid(noisy_ped) == 2
@test posf(noisy_ped).s == posf(ped_entity).s + noise(ped_entity).pos[1]
@test posf(noisy_ped).t == posf(ped_entity).t + noise(ped_entity).pos[2]
@test posf(noisy_ped).ϕ == posf(ped_entity).ϕ
@test vel(noisy_ped) == vel(ped_entity) + noise(ped_entity).vel
@test posg(noisy_ped).x == 24.9
@test posg(noisy_ped).y ≈ -9.7

noisy_veh = noisy_entity(bv_entity, ped_roadway)
@test laneid(noisy_veh) == 1
@test posf(noisy_veh).s == posf(bv_entity).s + noise(bv_entity).pos[1]
@test posf(noisy_veh).t == posf(bv_entity).t + noise(bv_entity).pos[2]
@test posf(noisy_veh).ϕ == posf(bv_entity).ϕ
@test vel(noisy_veh) == vel(bv_entity) + noise(bv_entity).vel
@test posg(noisy_veh).x ≈ 49.7
@test posg(noisy_veh).y ≈ -0.1

# Check the noisy_scene function
scene = Scene([ped_entity, bv_entity])
n_scene = noisy_scene(scene, ped_roadway)
@test n_scene[1] == noisy_ped
@test n_scene[2] == noisy_veh

## Check constructors

# Test the BlinkerVehicle Constructor
veh = BlinkerVehicle(roadway = Tint_roadway, lane = 5, s = 30., v = 20., id = 1, goals = Tint_goals[5], blinker = true)
state = VehicleState(VecSE2(polar(20,-π/2) + VecE2(1.5,0), π/2), 20.)
@test all(isapprox.(posg(veh), posg(state)))
@test vel(veh) == vel(state)
@test posf(veh) == posf(veh.state.veh_state)
@test blinker(veh)
@test goals(veh) == [5,6]
@test laneid(veh) == 5
@test can_have_goal(veh, 6, Tint_roadway)
@test can_have_goal(veh, 5, Tint_roadway)
@test !can_have_goal(veh, 1, Tint_roadway)
@test !can_have_goal(veh, 2, Tint_roadway)

# Test the pedestrian Constructor
ped = NoisyPedestrian(roadway = ped_roadway, lane = 2, s = 2., v = 2., id = 1)
@test posf(ped).s == 2.
@test posg(ped).y ≈ -8.
@test posg(ped).x == 25.
@test vel(ped) == 2.
@test laneid(ped) == 2


## Check the behavior of the actions for the Pedestrian Control
simple_acc = PedestrianControl(a = (0.1, 0.1))
acc_state = propagate(ped_entity, simple_acc, ped_roadway, 0.1)
@test vel(acc_state) > vel(ped_entity)
@test posf(acc_state).s > posf(ped_entity).s
@test posf(acc_state).t > posf(ped_entity).t
@test posf(acc_state).ϕ > posf(ped_entity).ϕ

test_action = PedestrianControl(a = (0.2, 0.2), da = (-0.1,-0.1), noise = neg_noise)
new_state = propagate(ped_entity, test_action, ped_roadway, 0.1)
new_ped = Entity(new_state, PEDESTRIAN_DEF, 1)
@test laneid(new_ped) == 2
@test vel(acc_state) == vel(new_state)
@test all(posg(acc_state) .== posg(new_state))
@test noise(new_state) == test_action.noise

## Check the behavior of actions for BlinkerVehicle
simple_acc = BlinkerVehicleControl(a = 0.1)
acc_state = propagate(veh, simple_acc, Tint_roadway, 0.1)
@test vel(acc_state) > vel(veh)
@test posf(acc_state).s > posf(veh).s
@test posf(acc_state).t == posf(veh).t
@test posf(acc_state).ϕ == posf(veh).ϕ

test_action = BlinkerVehicleControl(a = 0.2, da = -0.1, toggle_goal = true, toggle_blinker = true, noise = neg_noise)
new_state = propagate(veh, test_action, Tint_roadway, 0.1)
new_veh = Entity(new_state, VehicleDef(), 1)
@test laneid(new_veh) == 6
@test blinker(new_state) == false
@test vel(acc_state) == vel(new_state)
@test all(posg(acc_state) .== posg(new_state))
@test noise(new_state) == test_action.noise


## Test the models

# Test the contruction of an IDM Model
Tmodel = TIDM(  yields_way = Tint_yields_way,
                intersection_enter_loc = Tint_intersection_enter_loc,
                intersection_exit_loc = Tint_intersection_exit_loc,
                goals = Tint_goals,
                should_blink = Tint_should_blink
             )

@test Tmodel.idm isa IntelligentDriverModel
@test Tmodel.noisy_observations == false
test_action = BlinkerVehicleControl(a = 0.2, da = -0.1, toggle_goal = false, toggle_blinker = true, noise = neg_noise)
Tmodel.next_action = test_action
track_longitudinal!(Tmodel.idm, 15., 0., 10.)
actual_action = rand(Random.GLOBAL_RNG, Tmodel)
@test actual_action.a == Tmodel.idm.a
@test actual_action.da == test_action.da
@test actual_action.toggle_goal == test_action.toggle_goal
@test actual_action.toggle_blinker == test_action.toggle_blinker
@test actual_action.noise == test_action.noise

# Test the construction of an AdversarialPedestrian model
ped_model = AdversarialPedestrian()
@test ped_model.idm isa IntelligentDriverModel
ped_model.next_action = PedestrianControl(a = (0.2, 0.2), da = (-0.1,-0.1), noise = neg_noise)
track_longitudinal!(ped_model.idm, 15., 0., 10.)
actual_action = rand(Random.GLOBAL_RNG, ped_model)
@test actual_action.a[1] == ped_model.idm.a
@test actual_action.a[2] == 0.
@test all(isapprox.(actual_action.da, (-0.1,-0.1)))
@test actual_action.noise == neg_noise

## Test lane prediction
veh1 = BlinkerVehicle(roadway = Tint_roadway, lane = 2, s = 40., v = 20., id = 1, goals = Tint_goals[2], blinker = false)
veh2 = BlinkerVehicle(roadway = Tint_roadway, lane = 2, s = 40., v = 20., id = 1, goals = Tint_goals[2], blinker = true)
veh3 = BlinkerVehicle(roadway = Tint_roadway, lane = 2, s = 50., v = 20., id = 1, goals = Tint_goals[2], blinker = false)
veh4 = BlinkerVehicle(roadway = Tint_roadway, lane = 2, s = 50., v = 20., id = 1, goals = Tint_goals[2], blinker = true)

@test lane_belief(veh1, Tmodel, Tint_roadway) == 2
@test lane_belief(veh2, Tmodel, Tint_roadway) == 1
@test lane_belief(veh3, Tmodel, Tint_roadway) == 2
@test lane_belief(veh4, Tmodel, Tint_roadway) == 2

## Check collisions
egovehicle = BlinkerVehicle(roadway = Tint_roadway, lane = 5, s = 50., v = 9., id = 5, goals = Tint_goals[5], blinker = true)
egovehicle2 = BlinkerVehicle(roadway = Tint_roadway, lane = 5, s = 35., v = 9., id = 5, goals = Tint_goals[5], blinker = true)
vehicles = [BlinkerVehicle(roadway = Tint_roadway, lane = 4, s = 50., v = 20., id = 1, goals = Tint_goals[5], blinker = false),
            BlinkerVehicle(roadway = Tint_roadway, lane = 2, s = 50., v = 19., id = 2, goals = Tint_goals[5], blinker = false),
            BlinkerVehicle(roadway = Tint_roadway, lane = 2, s = 10., v = 19., id = 3, goals = Tint_goals[5], blinker = false),
            BlinkerVehicle(roadway = Tint_roadway, lane = 4, s = 40., v = 14., id = 4, goals = Tint_goals[5], blinker = true)]

scene_ego = Scene([egovehicle, vehicles[2:end]...])
scene_any = Scene([egovehicle2, vehicles...])

@test any_collides(scene_any)
@test any_collides(scene_ego)
@test ego_collides(5, scene_ego)
@test !ego_collides(3, scene_any)

egovehicle = BlinkerVehicle(roadway = ped_roadway, lane = 1, s = 25., v=20., id = 2, blinker = false, goals = ped_goals[1])
pedestrian = NoisyPedestrian(roadway = ped_roadway, lane = 2, s = 9.5, v = 2., id = 1)
scene = Scene([pedestrian, egovehicle])

@test any_collides(scene)
@test ego_collides(2, scene)

## Construct a whole scene and let it run (T-Intersection)
egovehicle = BlinkerVehicle(roadway = Tint_roadway, lane = 5, s = 35., v = 9., id = 5, goals = Tint_goals[5], blinker = true)
vehicles = [BlinkerVehicle(roadway = Tint_roadway, lane = 3, s = 30., v = 20., id = 1, goals = Tint_goals[5], blinker = false),
            BlinkerVehicle(roadway = Tint_roadway, lane = 2, s = 15., v = 19., id = 2, goals = Tint_goals[5], blinker = false),
            BlinkerVehicle(roadway = Tint_roadway, lane = 2, s = 10., v = 19., id = 3, goals = Tint_goals[5], blinker = false),
            BlinkerVehicle(roadway = Tint_roadway, lane = 4, s = 40., v = 14., id = 4, goals = Tint_goals[5], blinker = true)]

scene = Scene([egovehicle, vehicles...])

@test !any_collides(scene)
@test !ego_collides(5, scene)

models = Dict(i => TIDM(Tint_TIDM_template) for i=1:5)

nticks, timestep = 100, 0.1
scenes = AutomotiveSimulator.simulate(scene, Tint_roadway, models, nticks, timestep)
@test length(scenes) == nticks + 1

# test end_of_road
@test end_of_road(scenes[end][1], Tint_roadway, Inf)
@test end_of_road(scenes[end][2], Tint_roadway, Inf)
@test end_of_road(scenes[end][3], Tint_roadway, Inf)
@test end_of_road(scenes[end][4], Tint_roadway, Inf)
@test end_of_road(scenes[end][5], Tint_roadway, Inf)

egovehicle = BlinkerVehicle(roadway = Tint_roadway, lane = 5, s = 35., v = 9., id = 5, goals = Tint_goals[5], blinker = true)
@test end_of_road(egovehicle, Tint_roadway, 35)

# Note: use the following code to make a gif
# The cars are not removed at the end of the roadway, the mpd takes care of that
# using Reel
# animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
#     i = Int(floor(t/dt)) + 1
#     render([Tint_roadway, scenes[i]], canvas_width=1200, canvas_height=800)
# end
# write("Tint_roadway_animated.gif", animation)


## Construct a scene and let it run with the pedestrian and the car
pedestrian = NoisyPedestrian(roadway = ped_roadway, lane = 2, s = 5., v = 2., id = 1)
egovehicle = BlinkerVehicle(roadway = ped_roadway, lane = 1, s = 0., v=10., id = 2, blinker = false, goals = ped_goals[1])
scene = Scene([pedestrian, egovehicle])

models = Dict(1=>AdversarialPedestrian(idm = IntelligentDriverModel(v_des = 2)), 2 => ped_TIDM_template)

nticks = 50
timestep = 0.1
scenes = AutomotiveSimulator.simulate(scene, ped_roadway, models, nticks, timestep)
@test length(scenes) == nticks + 1

# animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
#     i = Int(floor(t/dt)) + 1
#     render([ped_roadway, crosswalk, scenes[i]], canvas_width=1200, canvas_height=800)
# end
# write("ped_roadway_animated.gif", animation)


## Run where noise causes the system to fail
pedestrian = NoisyPedestrian(roadway = ped_roadway, lane = 2, s = 5., v = 2., id = 1, noise = Noise(pos=(-100, 0), vel=-3.))
egovehicle = BlinkerVehicle(roadway = ped_roadway, lane = 1, s = 0., v=10., id = 2, blinker = false, goals = ped_goals[1])
scene = Scene([pedestrian, egovehicle])

vehicle_model = TIDM(ped_TIDM_template, noisy_observations = true)
ped_model = AdversarialPedestrian(
            idm = IntelligentDriverModel(v_des = 2),
            next_action = PedestrianControl(noise =  Noise(pos=(-100, 0)))
            )
models = Dict(1 => ped_model,  2 => vehicle_model)

nticks = 50
timestep = 0.1
scenes = AutomotiveSimulator.simulate(scene, ped_roadway, models, nticks, timestep)
@test length(scenes) == nticks + 1


# animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
#     i = Int(floor(t/dt)) + 1
#     render([ped_roadway, crosswalk, scenes[i]], canvas_width=1200, canvas_height=800)
# end
# write("ped_roadway_animated.gif", animation)


## Test that car tracks the object if it is outside the intersection
pedestrian = NoisyPedestrian(roadway = ped_roadway, lane = 2, s = 5., t=10., v = 2., id = 1, )
egovehicle = BlinkerVehicle(roadway = ped_roadway, lane = 1, s = 0., v=10., id = 2, blinker = false, goals = ped_goals[1])

scene = Scene([pedestrian, egovehicle])

vehicle_model = TIDM(ped_TIDM_template)
ped_model = AdversarialPedestrian(idm = IntelligentDriverModel(v_des = 2))
models = Dict(1 => ped_model,  2 => vehicle_model)

nticks = 50
timestep = 0.1
scenes = AutomotiveSimulator.simulate(scene, ped_roadway, models, nticks, timestep)
@test length(scenes) == nticks + 1


# animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
#     i = Int(floor(t/dt)) + 1
#     render([ped_roadway, crosswalk, scenes[i]], canvas_width=1200, canvas_height=800)
# end
# write("ped_roadway_animated.gif", animation)

## Test of the blindspot
@test !in_blindspot(VecSE2(5, 0, 0.), Blindspot(π/12., π/6), VecSE2(25, -3, π/2))
@test in_blindspot(VecSE2(5, 0, -π/6), Blindspot(π/12., π/6), VecSE2(25, -3, π/2))
@test !in_blindspot(VecSE2(5, 0, π/6), Blindspot(π/12., π/6), VecSE2(25, -3, π/2))
@test in_blindspot(VecSE2(5, 0, 0.), Blindspot(π/12., π/3), VecSE2(25, -3, π/2))


## Construct a disturbance model for the adversaries
disturbances = Sampleable[Normal(0,1), Bernoulli(0.5), Bernoulli(0.5), Bernoulli(0.), Bernoulli(0.)]
sut_agent = BlinkerVehicleAgent(rand_up_left(id=1, s_dist=Uniform(15.,25.), v_dist=Uniform(10., 29)), TIDM(Tint_TIDM_template))
right_adv = BlinkerVehicleAgent(rand_right(id=3, s_dist=Uniform(15.,35.), v_dist=Uniform(15., 29.)), TIDM(Tint_TIDM_template), disturbance_model = disturbances)
mdp = DrivingMDP(sut_agent, [right_adv], Tint_roadway, 0.2, γ = 0.95)

@test length(agents(mdp)) == 2
@test mdp.per_timestep_penalty == 0
@test mdp.v_des == 25

