# Appends curves to an existing curve
function append_to_curve!(target::Curve, newstuff::Curve)
    s_end = target[end].s
    for c in newstuff
        push!(target, CurvePt(c.pos, c.s+s_end, c.k, c.kd))
    end
    return target
end

# Get the lane object by index from the roadway
roadway_lane(roadway::Roadway, laneid::Int) = roadway.segments[laneid].lanes[1]

# Get the frenet position for provided lane id
function AutomotiveSimulator.Frenet(roadway::Roadway, laneid::Int, s::Float64, t::Float64 = 0., ϕ::Float64 = 0.)
    Frenet(roadway_lane(roadway, laneid), s, t, ϕ)
end

## Straight roadway with a crosswalk
struct Crosswalk
    crosswalk::Lane
end

# Render the crosswalk
function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, env::Crosswalk)
    curve = env.crosswalk.curve
    n = length(curve)
    pts = Array{Float64}(undef, 2, n)
    for (i,pt) in enumerate(curve)
        pts[:,i] .= pt.pos[1:2]
    end

    add_instruction!(
        rendermodel, render_dashed_line,
        (pts, colorant"white", env.crosswalk.width, 1.0, 1.0, 0.0, 0)
    )
    rendermodel
end

# The crosswalk is lane 2 in the function below
const DEFAULT_CROSSWALK_LANE = 2

function road_with_crosswalk(; roadway_length = 50., crosswalk_length = 20., crosswalk_width = 6.0, crosswalk_pos = roadway_length/2, )
    # Generate a straight 2-lane roadway and a crosswalk lane
    roadway = gen_straight_roadway(2, roadway_length)
    crosswalk_start = VecE2(crosswalk_pos, -crosswalk_length/2)
    crosswalk_end = VecE2(crosswalk_pos, crosswalk_length/2)
    crosswalk_lane = gen_straight_curve(crosswalk_start, crosswalk_end, 2)
    crosswalk = Lane(LaneTag(2,1), crosswalk_lane, width = crosswalk_width)
    cw_segment = RoadSegment(2, [crosswalk])
    push!(roadway.segments, cw_segment) # append it to the roadway


    # Describes which lanes each lane should yield to (i.e. lane 4 yields to 1 and 2)
    yields_way = Dict(1=>[2], 2=>[])

    # The entry point of the intersection for each lane
    intersection_enter_loc = Dict(
        1 => VecSE2(crosswalk_pos - crosswalk_width /2., 0., 0),
        2 => VecSE2(crosswalk_pos, -DEFAULT_LANE_WIDTH / 2., 0)
        )

    # the exit point of the intersection for each lane
    intersection_exit_loc = Dict(
        1 => VecSE2(crosswalk_pos + crosswalk_width /2., 0., 0),
        2 => VecSE2(crosswalk_pos, 3*DEFAULT_LANE_WIDTH / 2., 0)
        )

    goals = Dict{Int64, Array{Int}}(
        1 => [1],
        2 => [2],
        )

    should_blink = Dict{Int64, Bool}(
        1 => false,
        2 => false,
        )

    roadway, Crosswalk(crosswalk), yields_way, intersection_enter_loc, intersection_exit_loc, goals, should_blink
end

ped_roadway, crosswalk, ped_yields_way, ped_intersection_enter_loc, ped_intersection_exit_loc, ped_goals, ped_should_blink = road_with_crosswalk()

## T-Intersection Roadway

# LP is the left-most point in between upper and lower lanes
# RP is the Right-most point in between upper and lower lanes
# BP is the Bottom-most point in between left and right lanes
# C is the center of the intersection
# dx is half lane shift in the x direction
# dy is half lane shift in the y direction
# r is the turn radius for the intersection
function T_intersection(; LP = VecE2(-50,0), RP = VecE2(50,0), BP = VecE2(0,-50), C = VecE2(0,0), dx = VecE2(DEFAULT_LANE_WIDTH / 2, 0), dy = VecE2(0, DEFAULT_LANE_WIDTH / 2), r = 4.)

    # Setup the roadway
    roadway = Roadway()

    # Append right turn coming from the left
    curve = gen_straight_curve(LP - dy, C - dy - r*dx, 2)
    append_to_curve!(curve, gen_bezier_curve(VecSE2(C - dy - r*dx, 0), VecSE2(C - r*dy - dx, -π/2), 0.6r, 0.6r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(C - r*dy - dx, BP - dx, 2))
    lane1 = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane1.tag.segment, [lane1]))

    # Append straight right
    curve = gen_straight_curve(LP - dy, RP - dy, 2)
    lane2 = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane2.tag.segment, [lane2]))

    # Append straight left
    curve = gen_straight_curve(RP + dy, LP + dy, 2)
    lane3 = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane3.tag.segment, [lane3]))

    # Append left turn coming from the right
    curve = gen_straight_curve(RP + dy, C + r*dx + dy, 2)
    append_to_curve!(curve, gen_bezier_curve(VecSE2(C + r*dx + dy,-π), VecSE2(C - dx - r*dy, -π/2), r, r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(C - dx - r*dy, BP - dx, 2))
    lane4 = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane4.tag.segment, [lane4]))

    # Append left turn coming from below
    curve = gen_straight_curve(BP + dx, C + dx - r*dy, 2)
    append_to_curve!(curve, gen_bezier_curve(VecSE2(C + dx - r*dy, π/2), VecSE2(C - r*dx + dy, -π), r, r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(C - r*dx + dy, LP + dy, 2))
    lane5 = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane5.tag.segment, [lane5]))

    # Append right turn coming from below
    curve = gen_straight_curve(BP + dx, C + dx - r*dy, 2)
    append_to_curve!(curve, gen_bezier_curve(VecSE2(C + dx - r*dy, π/2), VecSE2(C + r*dx - dy, 0), 0.6r, 0.6r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(C + r*dx - dy, RP - dy, 2))
    lane6 = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane6.tag.segment, [lane6]))

    # Setup dictionaries that describe priorities, goals and intersection information
    ################################################################################

    # Describes which lanes each lane should yield to (i.e. lane 4 yields to 1 and 2)
    yields_way = Dict(1=>[], 2=>[], 3=>[], 4=>[1,2], 5=>[2,3,4], 6=>[1,2,])

    # The entry point of the intersection for each lane
    intersection_enter_loc = Dict(
        1=> VecSE2(C - dy - r*dx, 0),
        2=> VecSE2(C - dy - r*dx, 0),
        3=> VecSE2(C + dy + r*dx, π),
        4=> VecSE2(C + dy + r*dx, π),
        5=> VecSE2(C + dx - r*dy, π/2),
        6=> VecSE2(C + dx - r*dy, π/2),
        )

    # the exit point of the intersection for each lane
    intersection_exit_loc = Dict(
        1=> VecSE2(C - r*dy - dx, -π/2),
        2=> VecSE2(C - dy + r*dx, 0),
        3=> VecSE2(C + dy - r*dx, π),
        4=> VecSE2(C - r*dy - dx, -π/2),
        5=> VecSE2(C + dy - r*dx, π),
        6=> VecSE2(C - dy + r*dx, 0),
        )

    goals = Dict{Int64, Array{Int}}(
        1 => [1,2],
        2 => [1,2],
        3 => [3,4],
        4 => [3,4],
        5 => [5,6],
        6 => [5,6]
    )

    # Whether or not the blinker should be on for each lane
    should_blink = Dict{Int64, Bool}(
        1 => true,
        2 => false,
        3 => false,
        4 => true,
        5 => true,
        6 => true,
        )

    return roadway, yields_way, intersection_enter_loc, intersection_exit_loc, goals, should_blink
end

# Strings describing the goal of each lane
Tint_goal_label = Dict(1 => "turn right", 2 =>"straight", 3=>"straight", 4=>"turn left", 5=>"turn left", 6=> "turn right")

# whether the signal is on the right side
Tint_signal_right = Dict(
    1 => true,
    2 => true,
    3 => false,
    4 => false,
    5 => false,
    6 => true
    )

# Construct a global T-intersection to be used
Tint_roadway, Tint_yields_way, Tint_intersection_enter_loc, Tint_intersection_exit_loc, Tint_goals, Tint_should_blink = T_intersection()

# Construct template TIDM models (one for each roadway)

Tint_TIDM_template = TIDM(yields_way = Tint_yields_way,
                    intersection_enter_loc = Tint_intersection_enter_loc,
                    intersection_exit_loc = Tint_intersection_exit_loc,
                    goals = Tint_goals,
                    should_blink = Tint_should_blink
                    )


ped_TIDM_template = TIDM(yields_way = ped_yields_way,
                        intersection_enter_loc = ped_intersection_enter_loc,
                        intersection_exit_loc = ped_intersection_exit_loc,
                        goals = ped_goals,
                        should_blink = ped_should_blink
                        )
