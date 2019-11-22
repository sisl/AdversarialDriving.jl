using AutomotiveDrivingModels

# Appends curves to an existing curve
function append_to_curve!(target::Curve, newstuff::Curve)
    s_end = target[end].s
    for c in newstuff
        push!(target, CurvePt(c.pos, c.s+s_end, c.k, c.kd))
    end
    return target
end

# LP is the left-most point in between upper and lower lanes
# RP is the Right-most point in between upper and lower lanes
# BP is the Bottom-most point in between left and right lanes
# C is the center of the intersection
# dx is half lane shift in the x direction
# dy is half lane shift in the y direction
# r is the turn radius for the intersection
function generate_T_intersection(; LP = VecE2(-50,0), RP = VecE2(50,0), BP = VecE2(0,-50), C = VecE2(0,0), dx = VecE2(DEFAULT_LANE_WIDTH / 2, 0), dy = VecE2(0, DEFAULT_LANE_WIDTH / 2), r = 4.)

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

    return roadway, intersection_enter_loc, intersection_exit_loc, goals, should_blink
end






