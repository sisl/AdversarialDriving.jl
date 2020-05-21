using AdversarialDriving
using AutomotiveSimulator
using AutomotiveVisualization
using Random
using Test

# construct the roadway
roadway, yields_way, intersection_enter_loc, intersection_exit_loc, goals, should_blink, dx, dy = T_intersection()

# a = render([roadway])

# Construct a vehicle
veh = BlinkerVehicle(VecSE2(polar(20,-π/2) + dx, π/2), 20., goals[5], 5, true, 1, roadway)
state = VehicleState(VecSE2(polar(20,-π/2) + dx, π/2), 20.)

# Check that all the elements were constructed properly
@test all(isapprox.(posg(veh), posg(state)))
@test vel(veh) == vel(state)
@test posf(veh) == posf(veh.state.veh_state)
@test veh.state.blinker
@test veh.state.goals == [5,6]
@test laneid(veh) == 5
@test can_have_goal(veh, 6, roadway)
@test can_have_goal(veh, 5, roadway)
@test !can_have_goal(veh, 1, roadway)
@test !can_have_goal(veh, 2, roadway)

# Check the behavior of actions
simple_acc = LaneFollowingAccelBlinker(a = 0.1)
acc_state = propagate(veh, simple_acc, roadway, 0.1)

test_action = LaneFollowingAccelBlinker(a = 0.2, da = -0.1, toggle_goal = true, toggle_blinker = true)
new_state = propagate(veh, test_action, roadway, 0.1)
new_veh = Entity(new_state, VehicleDef(), 1)
@test laneid(new_veh) == 6
@test new_state.blinker == false
@test vel(acc_state) == vel(new_state)
@test all(posg(acc_state) .== posg(new_state))

# Construct a model
model = TIDM( yields_way = yields_way,
    intersection_enter_loc = intersection_enter_loc,
    intersection_exit_loc = intersection_exit_loc,
    goals = goals,
    should_blink = should_blink )

model.next_action = test_action
actual_action = rand(Random.GLOBAL_RNG, model)
@test actual_action.da == test_action.da
@test actual_action.toggle_goal == test_action.toggle_goal
@test actual_action.toggle_blinker == test_action.toggle_blinker

# Check lane prediction
veh1 = BlinkerVehicle(VecSE2(polar(10,-π) - dy, 0), 20., goals[2], 2, false, 1, roadway)
veh2 = BlinkerVehicle(VecSE2(polar(10,-π) - dy, 0), 20., goals[2], 2, true, 1, roadway)
veh3 = BlinkerVehicle(VecSE2(polar(0,-π) - dy, 0), 20., goals[2], 2, false, 1, roadway)
veh4 = BlinkerVehicle(VecSE2(polar(0,-π) - dy, 0), 20., goals[2], 2, true, 1, roadway)

@test lane_belief(veh1, model, roadway) == 2
@test lane_belief(veh2, model, roadway) == 1
@test lane_belief(veh3, model, roadway) == 2
@test lane_belief(veh4, model, roadway) == 2

## Construct a whole scene and let it run
egovehicle = BlinkerVehicle(VecSE2(polar(15.0,-π/2) + dx, π/2), 9., goals[5], 5, true, 5, roadway)

# create list of other vehicles and models
vehicles = [BlinkerVehicle(VecSE2(polar(20.0, 0) + dy, -π), 20., goals[3], 3, false, 1, roadway),
            BlinkerVehicle(VecSE2(polar(35.0,-π) - dy, 0), 19., goals[2], 2, false, 2, roadway),
            BlinkerVehicle(VecSE2(polar(40.0,-π) - dy, 0), 19., goals[2], 2, false, 3, roadway),
            BlinkerVehicle(VecSE2(polar(10.0, 0) + dy, -π), 14., goals[4], 4, true, 4, roadway),
            ]

scene = Scene([egovehicle, vehicles...])
models = Dict(i => TIDM(model) for i=1:5)

nticks = 100
timestep = 0.1
scenes = AutomotiveSimulator.simulate(scene, roadway, models, nticks, timestep)
@test length(scenes) == nticks + 1


# using Reel
# animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
#     i = Int(floor(t/dt)) + 1
#     render([roadway, scenes[i]], canvas_width=1200, canvas_height=800)
# end
# write("roadway_animated.gif", animation)

