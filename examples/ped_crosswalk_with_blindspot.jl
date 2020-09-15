using AdversarialDriving
using AutomotiveSimulator, AutomotiveVisualization
using POMDPs, POMDPPolicies, POMDPSimulators

blindspot = Blindspot(π/12 + 0.07, π/15)
sut_agent = BlinkerVehicleAgent(get_ped_vehicle(id=1, s=5., v=15.), TIDM(ped_TIDM_template, noisy_observations = true, blindspot = blindspot))
# adv_ped = NoisyPedestrianAgent(get_pedestrian(id=2, s=10.7, v=0.), AdversarialPedestrian(idm=IntelligentDriverModel(v_des = 0)))
adv_ped = NoisyPedestrianAgent(get_pedestrian(id=2, s=6.4, v=0.), AdversarialPedestrian(idm=IntelligentDriverModel(v_des = 0)))
mdp = AdversarialDrivingMDP(sut_agent, [adv_ped], ped_roadway, 0.1)

# Sample blindspot behavior
hist = POMDPSimulators.simulate(HistoryRecorder(max_steps = 25), mdp, FunctionPolicy((s) -> Disturbance[PedestrianControl()]))

# Make the renderable object to visualize the blindspot
rb = (s) -> RenderableBlindspot(posg(get_by_id(s, sutid(mdp))), sut(mdp).model.blindspot, 30, colorant"black")

s = initialstate(mdp)
c = render([mdp.roadway,
                crosswalk,
                rb(s),
                FancyCar(car = s[2], color = colorant"blue"),
                FancyPedestrian(ped = s[1])],
                surface=AutomotiveVisualization.CairoPDFSurface(IOBuffer(), DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT))
write("blindspot.pdf", c)


scenes_to_gif([s.s for s in hist], mdp.roadway, "blindspot.gif", others = [crosswalk], others_fn = [rb])

