using AdversarialDriving
using POMDPs, POMDPPolicies, POMDPSimulators
using AutomotiveVisualization

sut_agent = BlinkerVehicleAgent(get_ped_vehicle(id=1, s=5., v=15.), TIDM(ped_TIDM_template, noisy_observations = true))
adv_ped = NoisyPedestrianAgent(get_pedestrian(id=2, s=7., v=2.0), AdversarialPedestrian())
mdp = AdversarialDrivingMDP(sut_agent, [adv_ped], ped_roadway, 0.2)

# Render a single frame
s = rand(initialstate(mdp))
c = render([mdp.roadway,
                crosswalk,
                VelocityArrow(entity = s[1], color = colorant"black"),
                VelocityArrow(entity = s[2], color = colorant"black"),
                FancyCar(car = s[2], color = colorant"blue"),
                FancyPedestrian(ped = s[1])],
                surface=AutomotiveVisualization.CairoPDFSurface(IOBuffer(), DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT))
write("pedestrian_crosswalk.pdf", c)

null_action = Disturbance[PedestrianControl()]
noisy_action = Disturbance[PedestrianControl(noise = Noise((-10.,0.), -2))]

# Nominal Behavior
hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> null_action))
scenes_to_gif([s.s for s in hist], mdp.roadway, "ped_crosswalk_nominal.gif", others = [crosswalk])

# Behavior with noise
hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> noisy_action))
scenes_to_gif([s.s for s in hist], mdp.roadway, "ped_crosswalk_failure.gif", others = [crosswalk])

