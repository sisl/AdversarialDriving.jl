# This example shows the T-intersection scenario -- nominal and failure scenario
using AdversarialDriving
using POMDPs, POMDPPolicies, POMDPSimulators

sut_agent = BlinkerVehicleAgent(up_left(id=1, s=25., v=15.), TIDM(Tint_TIDM_template, noisy_observations = true))
adv1 = BlinkerVehicleAgent(left_straight(id=2, s=0., v=15.0), TIDM(Tint_TIDM_template))
adv2 = BlinkerVehicleAgent(left_turnright(id=3, s=10., v=15.0), TIDM(Tint_TIDM_template))
adv3 = BlinkerVehicleAgent(right_straight(id=4, s=40., v=20.0), TIDM(Tint_TIDM_template))
adv4 = BlinkerVehicleAgent(right_turnleft(id=5, s=30., v=20.0), TIDM(Tint_TIDM_template))

mdp = AdversarialDrivingMDP(sut_agent, [adv1, adv2, adv3, adv4], Tint_roadway, 0.1)

null_action = actions(mdp)[1]
blinker_action = actions(mdp)[7]

# Nominal Behavior
hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> null_action))
scenes_to_gif(state_hist(hist), mdp.roadway, "Tint_nominal.gif")

# Behavior with noise
blinker_pol = FunctionPolicy((s) -> !blinker(get_by_id(s, 2)) ? blinker_action : null_action)
hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, blinker_pol)
scenes_to_gif(state_hist(hist), mdp.roadway, "Tint_failure.gif")

