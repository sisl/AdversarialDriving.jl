using AdversarialDriving
using AutomotiveSimulator
using Test
using POMDPs

## Test pedestrian conversion functions
@test PEDESTRIAN_OBS == 4
vec_to_NoisyPedestrian = vec_to_NoisyPedestrian_fn(2)
ped = NoisyPedestrian(roadway = ped_roadway, lane = 2, s = 5., v = 2., id = 1, noise = Noise(vel=10))
pedvec = [25., -5, π/2., 2.]
@test all(NoisyPedestrian_to_vec(ped) .≈ pedvec)
ped_recovered = vec_to_NoisyPedestrian(pedvec, 1, ped_roadway, nothing)
@test posf(ped_recovered).s ≈ 5.
@test posf(ped_recovered).t ≈ 0.
@test posf(ped_recovered).ϕ ≈ 0.
@test vel(ped_recovered) ≈ 2.
@test ped_recovered.id == 1
@test laneid(ped_recovered) == 2
@test noise(ped_recovered) == Noise() # Note that the noise is not preserved


## Test blinker vehicle conversion functions
@test BLINKERVEHICLE_OBS == 4
@test BLINKERVEHICLE_EXPANDED_OBS == 30

bv = BlinkerVehicle(roadway = Tint_roadway, lane = 5, s = 35., v = 9., id = 5, goals = Tint_goals[5], noise = Noise(vel=10.), blinker = true)
bvvec = [35., 9., 5., 1.]
@test all(BlinkerVehicle_to_vec(bv) .≈ bvvec)
bv_recovered = vec_to_BlinkerVehicle(bvvec, 5, Tint_roadway, Tint_TIDM_template)
posf(bv_recovered).s == 35.
@test vel(bv_recovered) == 9.
@test laneid(bv_recovered) == 5
@test blinker(bv_recovered)

bv_expanded = BlinkerVehicle_to_expanded_vec(bv)
@test length(bv_expanded) == 30
@test sum(bv_expanded .== 0) ==  25
@test sum(bv_expanded) == sum(1 + 35 + 9 + 1  + 81) # lane, pos, vel, blinker, vel^2

## Test convert_s functions
bv1 = BlinkerVehicleAgent(up_left(id=10), TIDM(Tint_TIDM_template))
bv2 = BlinkerVehicleAgent(left_straight(id=2), TIDM(Tint_TIDM_template))
bv3 = BlinkerVehicleAgent(right_turnleft(id=3), TIDM(Tint_TIDM_template))
bv4 = BlinkerVehicleAgent(left_turnright(id=4, s=40.), TIDM(Tint_TIDM_template))
mdp = AdversarialDrivingMDP(bv1, [bv2, bv3, bv4], Tint_roadway, 0.1)

vec = convert_s(AbstractArray, initialstate(mdp), mdp)
s_back = convert_s(Scene, vec, mdp)
@test s_back[1] == initialstate(mdp)[1]
@test s_back[2] == initialstate(mdp)[2]
@test s_back[3] == initialstate(mdp)[3]
@test s_back[4] == initialstate(mdp)[4]


