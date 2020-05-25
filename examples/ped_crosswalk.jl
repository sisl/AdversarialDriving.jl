# This example shows the pedestrain and crosswalk scenario -- nominal and failure scenario
using AdversarialDriving
using POMDPs, POMDPPolicies, POMDPSimulators

sut_agent = BlinkerVehicleAgent(ez_ped_vehicle(id=1, s=5., v=15.), TIDM(ped_TIDM_template, noisy_observations = true))
adv_ped = NoisyPedestrianAgent(ez_pedestrian(id=2, s=7., v=2.0), AdversarialPedestrian())
mdp = AdversarialDrivingMDP(sut_agent, [adv_ped], ped_roadway, 0.1, discrete = false)

null_action = Disturbance[PedestrianControl()]
noisy_action = Disturbance[PedestrianControl(noise = Noise((-10.,0.), -2))]

# Nominal Behavior
hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> null_action))
scenes_to_gif(state_hist(hist), mdp.roadway, "ped_crosswalk_nominal.gif", others = [crosswalk])

# Behavior with noise
hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> noisy_action))
scenes_to_gif(state_hist(hist), mdp.roadway, "ped_crosswalk_failure.gif", others = [crosswalk])

