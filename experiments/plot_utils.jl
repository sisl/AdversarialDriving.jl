using Interact
using Reel
using AutoViz
using Plots
using Statistics

# Strings describing the goal of each lane
goal_map = Dict(1 => "turn right", 2 =>"straight", 3=>"straight", 4=>"turn left", 5=>"turn left", 6=> "turn right")

# whether the signal is on the right side
lane_right = Dict(
    1 => true,
    2 => true,
    3 => false,
    4 => false,
    5 => false,
    6 => true
    )

function write_scene(scene, models, roadway, filename; egoid = nothing, text = false)
    p = plot_scene(scene, models, roadway, egoid = egoid, text = text)
    write_to_svg(p, filename)
end

function write_scenes(scenes, models, roadway, filename_base; egoid = nothing, text = false)
    i = 1
    for s in scenes
        filename = (i < 10) ? string(filename_base, "_0", i, ".png") : string(filename_base, "_", i, ".png")
        p = plot_scene(s, models, roadway, egoid = egoid, text = text)
        AutoViz.write_to_png(p, filename)
        i = i+1
    end
end

function plot_scene(scene, models, roadway; egoid = nothing, text = false)
    car_colors = Dict{Int, Colorant}()
    cam = FitToContentCamera(.1)
    overlays = SceneOverlay[]
    for (index,veh) in enumerate(scene)
        i = veh.id
        li = laneid(veh)
        if egoid == nothing || i != egoid
            car_colors[i] = colorant"red"
        else
            car_colors[i] = colorant"blue"
        end
        push!(overlays, BlinkerOverlay(on = veh.state.blinker, veh = Vehicle(veh),right=lane_right[li]))
        text && push!(overlays, TextOverlay(pos = veh.state.veh_state.posG + VecE2(-3, 3), text = [ string("id: ", veh.id, ", goal: ", goal_map[li])], color = colorant"black", incameraframe=true))

    end
    AutoViz.render(Scene(scene), roadway, overlays, cam=cam, car_colors = car_colors)
end

function make_interact(scenes, models, roadway; egoid = nothing)
    # interactive visualization
    @manipulate for frame_index in 1 : length(scenes)
        plot_scene(scenes[frame_index], models, roadway, egoid=egoid)
    end
end

function make_video(scenes, models, roadway, filename; egoid = nothing)
    frames = Frames(MIME("image/png"), fps=10)
    # interactive visualization
    for frame_index in 1 : length(scenes)
        c = plot_scene(scenes[frame_index], models, roadway, egoid=egoid)
        push!(frames, c)
    end
    write(filename, frames)
end

function plot_training(training_log, filename)
    p1 = plot(training_log["return"], xlabel="Iterations", ylabel="Return", title = "Batch Returns vs Iteration", label="Average", size = (600,200))
    p1 = plot!(training_log["max_return"], label="Max")
    savefig(p1, string(filename, "_returns.pdf"))

    p3 = plot(training_log["kl"], xlabel="Iterations", ylabel="KL Divergence", title = "KL Divergence", label="", size = (600,200))
    savefig(p3, string(filename, "_kl.pdf"))
end


function plot_time_series(time_series_array, label, p = nothing)
    # First combine the arrays
    new_arrays = []
    N = length(time_series_array[1])
    for i=1:N
        arrai = []
        for s in time_series_array
            push!(arrai, s[i])
        end
        push!(new_arrays, arrai)
    end
    means = mean.(new_arrays)
    stds = std.(new_arrays)
    p == nothing ? plot(means, ribbon=stds, label=label) : plot!(p, means, ribbon=stds, label=label)
end

