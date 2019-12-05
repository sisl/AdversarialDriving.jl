using Gym

o_dim(env::EnvWrapper) = env._env.observation_space.shape[1]
a_dim(env::EnvWrapper) = env._env.action_space.shape[1]
max_steps(env::EnvWrapper) = env.max_episode_steps
dt(env::EnvWrapper) = 1

initialstate(env::EnvWrapper) = reset!(env)

discount(env::EnvWrapper) = 1.

function policy_rollout(env::EnvWrapper, policy, s0)
    # Setup vectors to store episode information
    Nmax, osz, asz = 10*env.max_episode_steps+1, o_dim(env), a_dim(env)
    observations = Array{Float64, 2}(undef, Nmax, osz)
    actions = Array{Float64, 2}(undef, Nmax, asz)
    rewards = Array{Float64}(undef, Nmax)

    # Setup initial state
    s = s0

    i, done = 0, false
    while !done
        i += 1
        observations[i, :] .= s
        a = policy(s)
        actions[i, :] .= a
        s, r, done, _ = step!(env, a)
        rewards[i] = r
        if i >= env.max_episode_steps
            break
        end
    end
    return view(observations, 1:i, :), view(actions, 1:i, :), view(rewards, 1:i)
end

